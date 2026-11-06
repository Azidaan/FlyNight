"""Native mutex-only probes with unique names; never import Torch or use GPU."""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import json
import sys
import threading
import uuid
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tracking.gpu_lock import GpuLock

out = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report-name', default='before_report.json')
report_path = (out/parser.parse_args().report_name).resolve()
assert report_path.parent == out.resolve()
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = ctypes.c_void_p
k.GetProcessHandleCount.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
k.GetHandleInformation.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
k.CloseHandle.argtypes = (ctypes.c_void_p,)
k.CreateEventW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
k.CreateEventW.restype = ctypes.c_void_p
k.ReleaseMutex.argtypes = (ctypes.c_void_p,)
prefix = 'Local\\CodexGpuLockAudit_'+uuid.uuid4().hex

def handle_count():
    count = ctypes.c_ulong()
    assert k.GetProcessHandleCount(k.GetCurrentProcess(), ctypes.byref(count))
    return count.value

def close_if_open(handle):
    flags = ctypes.c_ulong()
    if handle and k.GetHandleInformation(handle, ctypes.byref(flags)):
        assert k.CloseHandle(handle)
        return True
    return False

report = dict(source_sha256=hashlib.sha256((ROOT/'tracking/gpu_lock.py').read_bytes()).hexdigest(),
              scope='Named Win32 synchronization only; no Torch import, no GPU access; all names unique')
event = k.CreateEventW(None, False, False, prefix+'_collision')
assert event
try:
    body = False
    error = None
    try:
        with GpuLock(prefix+'_collision'):
            body = True
    except Exception as exc:
        error = repr(exc)
    report['create_failure'] = dict(protected_body_executed=body, error=error)
finally:
    close_if_open(event)

before = handle_count()
locks = [GpuLock(prefix+'_leak_'+str(i)) for i in range(100)]
handles = [lock.handle for lock in locks]
assert all(handles)
allocated = handle_count()
del locks
gc.collect()
after_gc = handle_count()
cleaned = sum(close_if_open(handle) for handle in handles)
report['handles'] = dict(before=before, allocated=allocated, after_gc=after_gc,
                         reclaimed_by_probe=cleaned, after_cleanup=handle_count())

lock = GpuLock(prefix+'_ownership')
lock.__enter__()
release = {}
def wrong_thread_release():
    try:
        release['returned'] = lock.__exit__(None, None, None)
        release['error'] = None
    except Exception as exc:
        release['error'] = repr(exc)
worker = threading.Thread(target=wrong_thread_release)
worker.start(); worker.join(timeout=5)
assert not worker.is_alive()
report['wrong_thread_release'] = release
# The creating/acquiring thread is still ours; release and close its handle.
assert k.ReleaseMutex(lock.handle)
close_if_open(lock.handle)
lock.handle = None
report['cleanup_complete'] = True

callers = []
for directory in ('tracking', 'scripts'):
    for path in (ROOT/directory).rglob('*.py'):
        source = path.read_text(encoding='utf-8-sig')
        if 'GpuLock' not in source and 'FOVEA_GPU_LOCK' not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        locks = []
        spawns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                if function.endswith('GpuLock'):
                    locks.append(dict(line=node.lineno, argument=ast.unparse(node.args[0]) if node.args else 'omitted'))
                if function in ('subprocess.Popen', 'subprocess.run', 'subprocess.check_call', 'subprocess.check_output', 'Popen'):
                    spawns.append(dict(line=node.lineno, function=function))
        if locks or 'FOVEA_GPU_LOCK' in source:
            callers.append(dict(path=path.relative_to(ROOT).as_posix(), locks=locks, subprocess_calls=spawns,
                                environment_lock='FOVEA_GPU_LOCK' in source))
report['caller_inventory'] = callers
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key != 'caller_inventory'}, ensure_ascii=False, indent=2))
print('caller_files', len(callers))
