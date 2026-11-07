from pathlib import Path
import json
from scripts import b123_deadline_guard_recovery1_20261027 as guard


def test_atomic_status_write_retries_windows_sharing_failure(tmp_path,monkeypatch):
    original=Path.replace;attempts=[]
    def replace(path,target):
        attempts.append(target)
        if len(attempts)==1:raise PermissionError('temporary Windows sharing violation')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    target=tmp_path/'status.json'
    assert guard.write(target,{'state':'watching'})
    assert json.loads(target.read_text())=={'state':'watching'} and len(attempts)==2


def test_persistent_status_write_failure_does_not_raise_and_break_guard(tmp_path,monkeypatch):
    attempts=[]
    def replace(path,target):
        attempts.append(target);raise PermissionError('still locked')
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    assert guard.write(tmp_path/'status.json',{'state':'watching'}) is False
    assert len(attempts)==6
