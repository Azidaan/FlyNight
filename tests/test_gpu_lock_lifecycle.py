"""Named-mutex contract only; these tests never initialize CUDA."""
import ctypes
import gc
import sys
import threading
import time
import subprocess
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from tracking.gpu_lock import GpuLock

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows named mutex')


def kernel():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateEventW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
    k.CreateEventW.restype = ctypes.c_void_p
    k.GetHandleInformation.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    k.CloseHandle.argtypes = (ctypes.c_void_p,)
    k.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    k.WaitForSingleObject.restype = ctypes.c_uint32
    k.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    return k


def name():
    return 'Local\\CodexMutexTest_'+uuid.uuid4().hex


def close_native(lock):
    if lock.handle:
        assert kernel().CloseHandle(lock.handle)
        lock.handle = None


def test_create_failure_cannot_run_protected_body():
    k, key = kernel(), name()
    event = k.CreateEventW(None, False, False, key)
    assert event
    try:
        with pytest.raises(OSError):
            with GpuLock(key):
                pytest.fail('Unprotected body was reached after CreateMutexW failed')
    finally:
        assert k.CloseHandle(event)


def test_reusable_nested_context_and_explicit_close():
    lock = GpuLock(name())
    try:
        original = lock.handle
        with lock:
            with lock:
                with pytest.raises(RuntimeError, match='acquired'):
                    lock.close()
        with lock:
            assert lock.handle == original
        assert lock.waits == 3
        lock.close()
        assert lock.handle is None
        lock.close()
        with pytest.raises(RuntimeError, match='closed'):
            with lock:
                pass
    finally:
        close_native(lock)


def test_idle_discarded_locks_release_their_native_handles():
    k = kernel()
    locks = [GpuLock(name()) for _ in range(30)]
    handles = [lock.handle for lock in locks]
    assert all(handles)
    del locks
    gc.collect()
    remaining = []
    for handle in handles:
        flags = ctypes.c_ulong()
        if k.GetHandleInformation(handle, ctypes.byref(flags)):
            remaining.append(handle)
    for handle in remaining:
        assert k.CloseHandle(handle)
    assert remaining == []


def test_wrong_thread_cannot_silently_release_lock():
    lock = GpuLock(name())
    lock.__enter__()
    errors = []
    def wrong_thread():
        try:
            lock.__exit__(None, None, None)
        except (OSError, RuntimeError) as error:
            errors.append(error)
    thread = threading.Thread(target=wrong_thread)
    try:
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 1
    finally:
        lock.__exit__(None, None, None)
        close_native(lock)


def test_same_name_excludes_other_thread_until_all_nested_releases():
    key, k = name(), kernel()
    lock, other = GpuLock(key), GpuLock(key)
    results = []
    def probe():
        code = k.WaitForSingleObject(other.handle, 0)
        results.append(code)
        if code in (0, 0x80):
            assert k.ReleaseMutex(other.handle)
    def once():
        thread = threading.Thread(target=probe)
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive()
    try:
        with lock:
            with lock:
                once()
            once()
        once()
        assert results == [0x102, 0x102, 0]
    finally:
        close_native(lock); close_native(other)


def test_disabled_lock_preserves_noop_contract():
    lock = GpuLock(None)
    with lock:
        with lock:
            pass
    assert lock.handle is None and lock.waits == 0
    lock.close()


def test_same_instance_serializes_threads_and_can_close_after_waiters():
    lock = GpuLock(name())
    active = []
    errors = []
    def work():
        try:
            for _ in range(8):
                with lock:
                    active.append(threading.get_ident())
                    assert len(active) == 1
                    time.sleep(.001)
                    active.pop()
        except BaseException as error:
            errors.append(error)
    workers = [threading.Thread(target=work) for _ in range(3)]
    try:
        for worker in workers:worker.start()
        for worker in workers:worker.join(timeout=5)
        assert all(not worker.is_alive() for worker in workers)
        assert not errors and not active and lock.waits == 24
        lock.close()
    finally:
        close_native(lock)


def test_abandoned_owner_is_recovered_without_stale_recursion_depth():
    lock = GpuLock(name())
    worker = threading.Thread(target=lock.__enter__)
    try:
        worker.start(); worker.join(timeout=5)
        assert not worker.is_alive()
        with lock:
            pass
        lock.close()
        assert lock.handle is None and lock.waits == 2
    finally:
        close_native(lock)


@pytest.mark.parametrize('operation', ['WaitForSingleObject', 'ReleaseMutex', 'CloseHandle'])
def test_native_failure_is_explicit_and_preserves_recoverable_handle(operation):
    lock = GpuLock(name())
    original = lock._k
    fake = SimpleNamespace(**{key:getattr(original,key) for key in
        ('WaitForSingleObject','ReleaseMutex','CloseHandle')})
    setattr(fake, operation, lambda *args: 0xFFFFFFFF if operation=='WaitForSingleObject' else 0)
    acquired = operation == 'ReleaseMutex'
    if acquired:lock.__enter__()
    lock._k = fake
    try:
        with pytest.raises(OSError, match=operation):
            if operation == 'WaitForSingleObject':lock.__enter__()
            elif operation == 'ReleaseMutex':lock.__exit__(None,None,None)
            else:lock.close()
        assert lock.handle is not None
    finally:
        lock._k = original
        if acquired:lock.__exit__(None,None,None)
        lock.close()


@pytest.mark.parametrize('same_name', [True, False])
def test_named_mutex_coordinates_processes_only_for_same_name(tmp_path, same_name):
    key = name()
    ready, entered = tmp_path/'ready', tmp_path/'entered'
    root = Path(__file__).resolve().parents[1]
    child_code = (
        'from pathlib import Path; import sys; sys.path.insert(0,sys.argv[1]); '
        'from tracking.gpu_lock import GpuLock; '
        'Path(sys.argv[3]).write_text("ready"); '
        'lock=GpuLock(sys.argv[2]); lock.__enter__(); '
        'Path(sys.argv[4]).write_text("entered"); lock.__exit__(None,None,None); lock.close()'
    )
    parent = GpuLock(key)
    process = None
    try:
        with parent:
            process = subprocess.Popen([sys.executable, '-c', child_code, str(root),
                key if same_name else key+'_different', str(ready), str(entered)],
                creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic()+5
            while not ready.exists() and time.monotonic()<deadline and process.poll() is None:
                time.sleep(.01)
            assert ready.exists()
            if same_name:
                time.sleep(.05)
                assert not entered.exists() and process.poll() is None
            else:
                _, error = process.communicate(timeout=5)
                assert process.returncode == 0, error
                assert entered.exists()
        _, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
        assert entered.exists()
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
        parent.close()
