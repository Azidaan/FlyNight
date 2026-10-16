"""Read existing frame requests only; never invokes a model or modifies outputs."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
video = '00067cfb-e535423e'
paths = {
    'B': ROOT / 'runs/z13_triton_hota_v1_20261014/full/B/signatures' / video / 'signatures.npz',
    'D': ROOT / 'runs/z13_cache_speed_v1_20261014/D/triton' / video / 'signatures.npz',
}
requests = {}
frame_counts = {}
for arm, path in paths.items():
    with np.load(path) as archive:
        requests[arm] = set(archive['start'].tolist()) | set(archive['end'].tolist())
        frame_counts[arm] = int(archive['frame_count'])
assert len(set(frame_counts.values())) == 1
separate = sum(len(s) for s in requests.values())
union = len(requests['B'] | requests['D'])
overlap = len(requests['B'] & requests['D'])
proof = {'video': video, 'source_paths': {k: str(p.relative_to(ROOT)) for k, p in paths.items()},
         'frame_counts': frame_counts, 'needed_frames': {k: len(s) for k, s in requests.items()},
         'separate_feature_calls_B_D': separate, 'shared_feature_calls_B_D': union,
         'overlapping_frames_B_D': overlap, 'redundant_call_fraction_B_D': overlap / separate,
         'scope': 'One existing B/D video, no C result, no GPU or timing run. Frame features independent of requested boxes; this counts removable calls, not guaranteed elapsed-time reduction.'}
out = ROOT / 'runs/z13_masa_overlap_readonly_20261015.json'
out.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
