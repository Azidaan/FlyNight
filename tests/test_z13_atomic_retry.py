import json
import os
import threading
import ctypes

import pytest
from tracking.z13_checkpoint import atomic_json, replace_with_retry


def test_transient_denial_preserves_old_until_atomic_success(tmp_path, monkeypatch):
    dst = tmp_path / 'status.json'; dst.write_text('{"step": 1}')
    original = os.replace
    calls = []
    def flaky(source, destination):
        calls.append(1)
        if len(calls) < 3:
            assert json.loads(dst.read_text()) == {'step': 1}
            raise PermissionError('temporary sharing conflict')
        return original(source, destination)
    monkeypatch.setattr(os, 'replace', flaky)
    monkeypatch.setattr('tracking.z13_checkpoint.time.sleep', lambda _: None)
    atomic_json(dst, {'step': 2})
    assert len(calls) == 3 and json.loads(dst.read_text()) == {'step': 2}


def test_persistent_denial_is_bounded_and_keeps_both_files(tmp_path, monkeypatch):
    dst = tmp_path / 'state'; dst.write_text('old')
    src = tmp_path / 'next'; src.write_text('new')
    def blocked(*_):
        raise PermissionError('permanent')
    monkeypatch.setattr(os, 'replace', blocked)
    with pytest.raises(PermissionError):
        replace_with_retry(src, dst, attempts=2, delay=0)
    assert dst.read_text() == 'old' and src.read_text() == 'new'


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing semantics')
def test_real_windows_reader_without_delete_share(tmp_path):
    dst = tmp_path / 'status.json'; dst.write_text('{"step": 1}')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(dst), 0x80000000, 1, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    closed = threading.Event()
    def release():
        kernel.CloseHandle(handle); closed.set()
    timer = threading.Timer(.4, release); timer.start()
    try:
        atomic_json(dst, {'step': 2})
        assert closed.is_set() and json.loads(dst.read_text()) == {'step': 2}
    finally:
        timer.join()
