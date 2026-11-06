"""Small native K1 geometry/gradient check; no optimizer or training run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys, hashlib, json
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from scripts.fly_finetune_k1 import Fly, to_px
from tracking.flyvis_motion import TileLayout

report_path = OUT / 'probe_k1_report.json'
if report_path.exists():
    raise FileExistsError(report_path)
fly = Fly(30., layout=TileLayout(1, 1, 0.), gain=1., device='cpu')
fly.base.fade_in_seconds = .1
frames = np.random.default_rng(117).integers(0, 256, (2, 64, 96), dtype=np.uint8)
def parameter_hash():
    digest = hashlib.sha256()
    for component in (fly.net, fly.base.decoder):
        for name, value in sorted(component.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
before = parameter_hash()
with torch.no_grad():
    initial = fly.render(torch.from_numpy(fly.tiles(frames[0])).float().unsqueeze(1) / 255.)
    state = fly.fade_in(initial)
    baseline = fly.base.decoder(state.nodes.activity[:, None])[:, 0]
    current = fly.render(torch.from_numpy(fly.tiles(frames[1])).float().unsqueeze(1) / 255.)
    _, activity = fly.step(state, initial, current)
activity = activity.detach().requires_grad_(True)
raw = fly.base.decoder(activity).mean(1) - baseline
arrays = {'frames': frames}
comparisons = {}
for version in ('legacy_v1', 'resize_v2'):
    fly.base.geometry_version = version
    points, scales = fly.points_scale()
    converted = to_px(raw, torch.from_numpy(scales), torch.tensor(0.))
    expected = fly.base._points_and_vectors(raw)
    error = float(np.max(np.abs(converted.detach().numpy() - expected.vectors_xy)))
    assert error <= 1e-6 and np.array_equal(points, expected.points_xy)
    arrays[version + '_points'] = points
    arrays[version + '_vectors'] = converted.detach().numpy()
    comparisons[version] = {'max_abs_error': error, 'points_exact': True}
loss = converted.square().mean()
loss.backward()
assert torch.isfinite(activity.grad).all() and bool(torch.any(activity.grad != 0))
assert before == parameter_hash()
assert not torch.cuda.is_initialized()
np.savez_compressed(OUT / 'probe_k1_outputs.npz', **arrays)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(passed=True, comparisons=comparisons,
              decoder_input_gradient_finite=True, decoder_input_gradient_nonzero=True,
              parameters_unchanged=True, parameter_sha256=before, cuda_initialized=False,
              scope='2 synthetic frames, one tile, CPU; no optimizer, no dataset or HOTA',
              sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                  ROOT/'scripts/fly_finetune_k1.py', ROOT/'tracking/flyvis_motion.py',
                  ROOT/'tracking/flyvis_checkpoint.py', ROOT/'tracking/flyvis_adapter.py',
                  OUT/'probe_k1_outputs.npz']})
report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
