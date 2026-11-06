"""A denied taskkill must not silently leave the owned direct child running."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import z13_window_runner as runner


@pytest.mark.skipif(os.name != 'nt', reason='Windows owned Popen handle')
def test_denied_tree_kill_stops_only_owned_child_and_reports_limit(monkeypatch):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    own = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], creationflags=flags)
    sibling = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], creationflags=flags)
    calls = []
    def deny(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout=b'', stderr=b'Access denied')
    monkeypatch.setattr(runner.subprocess, 'run', deny)
    try:
        report = runner.stop_owned_child(own)
        assert report['pid'] == own.pid and report['exit_code'] is not None
        assert report['handle_fallback'] and report['tree_termination'] == 'unconfirmed'
        assert report['taskkill_returncode'] == 1 and report['taskkill_stderr'] == 'Access denied'
        assert own.poll() is not None and sibling.poll() is None
        assert calls == [['taskkill', '/PID', str(own.pid), '/T', '/F']]
    finally:
        for process in (own, sibling):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


def test_failed_handle_termination_is_explicit(monkeypatch):
    class Owned:
        pid = 123
        def poll(self):
            return None
        def terminate(self):
            raise PermissionError('handle denied')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout=b'', stderr=b'tree denied'))
    with pytest.raises(RuntimeError, match='Cannot stop owned child'):
        runner.stop_owned_child(Owned())


def test_exited_child_is_not_killed_by_reused_number(monkeypatch):
    class Finished:
        pid = 123
        returncode = 0
        def poll(self):
            return self.returncode
    def forbidden(*a, **kw):
        raise AssertionError('An exited Popen must never invoke taskkill')
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    assert runner.stop_owned_child(Finished()) == dict(pid=123, was_running=False,
        tree_termination='not_attempted', handle_fallback=False, exit_code=0)
