"""GPU lock names only: fake Win32 calls or fresh isolated mutex namespaces."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

import tracking.gpu_lock as locks

ALIASES = ('flyvis_fovea_gpu', 'Local\\DissertationSingleGpu',
           'FlyVis_z13_observed_memory_single_gpu')


@pytest.mark.parametrize('alias', ALIASES)
def test_known_names_use_one_native_mutex(monkeypatch, alias):
    created = []
    def create(security, owner, name):
        created.append(name)
        return 123
    fake = SimpleNamespace(CreateMutexW=create, WaitForSingleObject=lambda *a: 0,
                           ReleaseMutex=lambda *a: 1, CloseHandle=lambda *a: 1)
    monkeypatch.setattr(locks.sys, 'platform', 'win32')
    monkeypatch.setattr(locks.ctypes, 'WinDLL', lambda *a, **kw: fake, raising=False)
    lock = locks.GpuLock(alias)
    try:
        with lock:
            pass
        assert created == ['Local\\DissertationSingleGpu']
    finally:
        lock.close()


@pytest.mark.parametrize('value', [None, '', 'custom', 'flyvis_FOVEA_gpu',
                                  'Global\\DissertationSingleGpu'])
def test_disabled_and_custom_names_are_unchanged(value):
    assert locks.resolve_gpu_lock_name(value) == value


def test_environment_absence_keeps_lock_disabled(monkeypatch):
    monkeypatch.delenv('FOVEA_GPU_LOCK', raising=False)
    lock = locks.from_env()
    with lock:
        assert lock.handle is None and lock.name is None and lock.effective_name is None
    lock.close()


def unique():
    return 'Local\\CodexAliasTest_'+uuid.uuid4().hex


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows named mutex')
def test_nested_environment_alias_keeps_recursive_eyebank_contract(monkeypatch):
    key = unique()
    monkeypatch.setattr(locks, 'CANONICAL_GPU_LOCK_NAME', key)
    monkeypatch.setenv('FOVEA_GPU_LOCK', ALIASES[1])
    outer, bank, other = locks.GpuLock(ALIASES[0]), locks.from_env(), locks.GpuLock(ALIASES[2])
    results = []
    def probe():
        code = other._k.WaitForSingleObject(other.handle, 0)
        results.append(code)
        if code in (0, 0x80):
            assert other._k.ReleaseMutex(other.handle)
    def once():
        thread = threading.Thread(target=probe)
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive()
    try:
        assert outer.name == ALIASES[0] and bank.name == ALIASES[1]
        assert outer.effective_name == bank.effective_name == other.effective_name == key
        with outer:
            for _ in range(2):
                with bank:
                    once()
                once()
        once()
        assert results == [0x102, 0x102, 0x102, 0x102, 0]
        assert bank.waits == 2 and outer.waits == 1
    finally:
        outer.close(); bank.close(); other.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows named mutex')
@pytest.mark.parametrize('alias', ALIASES+(None,))
def test_cross_alias_process_coordination_uses_only_test_namespace(tmp_path, monkeypatch, alias):
    key = unique()
    monkeypatch.setattr(locks, 'CANONICAL_GPU_LOCK_NAME', key)
    ready, entered = tmp_path/'ready.json', tmp_path/'entered'
    child_name = alias if alias is not None else key+'_custom'
    child_code = '''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import tracking.gpu_lock as locks
locks.CANONICAL_GPU_LOCK_NAME=sys.argv[2]
lock=locks.GpuLock(sys.argv[3])
code=lock._k.WaitForSingleObject(lock.handle,0)
if code in (0,0x80):
    assert lock._k.ReleaseMutex(lock.handle)
Path(sys.argv[4]).write_text(json.dumps(dict(code=code,name=lock.effective_name)))
with lock:
    Path(sys.argv[5]).write_text('entered')
lock.close()
'''
    root = Path(__file__).resolve().parents[1]
    parent, process = locks.GpuLock(ALIASES[0]), None
    try:
        with parent:
            process = subprocess.Popen([sys.executable, '-c', child_code, str(root), key,
                child_name, str(ready), str(entered)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            deadline = time.monotonic()+5
            while not ready.exists() and time.monotonic()<deadline and process.poll() is None:
                time.sleep(.01)
            assert ready.exists()
            observed = json.loads(ready.read_text())
            if alias is not None:
                assert observed == dict(code=0x102, name=key)
                assert not entered.exists() and process.poll() is None
            else:
                _, error = process.communicate(timeout=5)
                assert process.returncode == 0, error
                assert observed == dict(code=0, name=child_name) and entered.exists()
        _, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
        assert entered.exists()
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
        parent.close()
