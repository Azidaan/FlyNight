"""Verify isolation and prepare resumption of the unchanged approved HOTA run."""
from datetime import datetime
from pathlib import Path
import ctypes
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json
PROBE = ROOT / 'runs/z13_cache_speed_v1_20261014'
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014/full'


def main():
    status = json.loads((PROBE / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] in ('completed', 'failed', 'paused'), 'Probe is still running'
    processes = WindowsProcesses()
    proof = json.loads((PROBE / 'launch_process.json').read_text(encoding='utf-8'))
    deadline = time.monotonic() + 30
    while processes.creation(proof['launcher_pid']) == int(proof['launcher_start_filetime']):
        assert time.monotonic() < deadline, 'Probe process has not exited'
        time.sleep(.5)
    full_status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    assert full_status['state'] == 'paused'
    authorization = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    assert datetime.now().astimezone() < datetime.fromisoformat(authorization['deadline'])
    records = list(RUN.glob('deadline_guard_*/result.json'))
    assert records
    previous = json.loads(max(records, key=lambda p: p.stat().st_mtime).read_text(encoding='utf-8'))
    assert not any(processes.creation(int(pid)) == created
                   for pid, created in previous['known_owned_processes'].items())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, 'flyvis_fovea_gpu')
    result = kernel.WaitForSingleObject(handle, 0)
    assert result in (0, 128), 'Another GPU process owns the mutex'
    kernel.ReleaseMutex(handle); kernel.CloseHandle(handle)
    stop = (RUN / 'stop.request').resolve(); stop.relative_to(ROOT)
    expected = 'Bounded author-requested speed comparison; completed videos preserved; resume same backend afterward.'
    assert stop.read_text(encoding='utf-8-sig').strip() == expected, 'New or unknown stop request; do not override it'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive = stop.with_name('stop.cache_probe_' + stamp + '.request')
    assert not archive.exists()
    stop.rename(archive)
    launch = RUN / 'launch_process.json'
    if launch.exists():
        old_launch = launch.with_name('launch_before_cache_probe_' + stamp + '.json')
        assert not old_launch.exists(); old_launch.write_bytes(launch.read_bytes())
    verification = {'verified': datetime.now().astimezone().isoformat(),
                    'probe_exited': True, 'old_hota_tree_exited': True, 'gpu_mutex_free': True,
                    'completed_jobs': full_status['completed_jobs'], 'deadline': authorization['deadline'],
                    'same_frozen_backend': True, 'stop_request_preserved': str(archive.relative_to(ROOT))}
    atomic_json(PROBE / 'resume_verified.json', verification)
    atomic_json(RUN / 'status.json', {'state': 'authorized', 'completed_jobs': full_status['completed_jobs'],
                                    'total_jobs': full_status['total_jobs'], 'reason': 'Resume unchanged backend after bounded speed probe'})
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
