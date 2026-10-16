"""Check frozen inputs on CPU without initializing CUDA or running the probe."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_train import Episodes
RUN = ROOT / 'runs/z13_cache_balanced_v1_20261015'
protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
data = Episodes(protocol['train_videos'])
episodes = list(dict.fromkeys(row[0] for row in data.labels if not row[8]))[:4]
assert len(episodes) == 4
assert all(data.n_frames(e) > 4 for e in episodes)
proof = {'gpu_initialized': False, 'source_videos': protocol['train_videos'],
         'episode_ids': episodes, 'frames': [data.n_frames(e) for e in episodes],
         'files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
             'scripts/z13_cache_balanced_probe.py', 'tracking/z13_cached_params_backend.py',
             'tracking/z13_compiled_backend.py', 'runs/z13_spatial_hota_v1_20261014/D/model.pt')},
         'syntax_checked': True, 'gpu_probe_run': False}
assert proof['files']['scripts/z13_cache_balanced_probe.py'] == protocol['probe_sha256']
(RUN / 'readiness.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
