"""Record the direct five-minute approval only after the own HOTA tree exits."""
from datetime import datetime
from pathlib import Path
import ctypes
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_cache_balanced_v1_20261015'
FULL = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
status = json.loads((FULL / 'status.json').read_text(encoding='utf-8'))
assert status['state'] == 'paused'
records = sorted(FULL.glob('deadline_guard_*/result.json'), key=lambda p: p.stat().st_mtime)
last = json.loads(records[-1].read_text(encoding='utf-8'))
native = WindowsProcesses()
assert not any(native.creation(int(pid)) == stamp for pid, stamp in last['known_owned_processes'].items())
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.CreateMutexW.restype = ctypes.c_void_p
k.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
k.ReleaseMutex.argtypes = (ctypes.c_void_p,)
k.CloseHandle.argtypes = (ctypes.c_void_p,)
h = k.CreateMutexW(None, False, 'flyvis_fovea_gpu')
assert k.WaitForSingleObject(h, 0) in (0, 128)
k.ReleaseMutex(h); k.CloseHandle(h)
now = datetime.now().astimezone()
proof = {'verified': now.isoformat(), 'own_hota_tree_exited': True, 'gpu_mutex_free': True,
         'completed_jobs_preserved': status['completed_jobs']}
atomic_json(RUN / 'stop_verified.json', proof)
atomic_json(RUN / 'status.json', {'state': 'authorized'})
path = ROOT / 'experiments/project_history.json'
catalog = json.loads(path.read_text(encoding='utf-8'))
card = next(c for c in catalog['experiments'] if c['id'] == 'z13_cache_balanced_v1')
assert card['state'] == 'planned'
card['state'] = 'running'
card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_cache_balanced_v1_20261015/stop_verified.json',
                         'claim': '19 готовых заданий сохранены, собственное дерево HOTA вышло, GPU освобождён'})
path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
