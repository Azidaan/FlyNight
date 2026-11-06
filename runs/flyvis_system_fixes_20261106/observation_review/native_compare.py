"""Run only after root marks the transaction implementation ready."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import json
import importlib.util
import hashlib
import time
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator

if (OUT / 'native_report.json').exists():
    raise FileExistsError('Preserve existing native report; create a separately named revision')
spec = importlib.util.spec_from_file_location('observation_before_runtime', OUT / 'before_flyvis_motion.py')
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
start = time.monotonic()
kwargs = dict(device='cpu', source_fps=30., dt=.02, flow_gain=1., geometry_version='legacy_v1')
old, new = before.FlyVisMotionEstimator(**kwargs), FlyVisMotionEstimator(**kwargs)
assert next(old.network.parameters()).device.type == next(new.network.parameters()).device.type == 'cpu'
assert not torch.cuda.is_initialized()
rng = np.random.default_rng(7001)
base = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
frames = [np.roll(np.roll(base, i, axis=0), 2*i, axis=1) for i in range(12)]
np.savez_compressed(OUT / 'native_inputs.npz', frames=np.stack(frames))
arrays = {}
checks = []

def exact_tensor(a, b):
    assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape
    assert torch.equal(a, b)

for mode in ('full', 'tiled'):
    for temporal in ('hold', 'linear'):
        for model in (old, new):
            model.mode = mode
            model.temporal_resampling = temporal
            model.reset()
        old_count, new_count = old.stats['observations'], new.stats['observations']
        for index, frame in enumerate(frames):
            a, b = old.observe(frame), new.observe(frame)
            np.testing.assert_array_equal(a.points_xy, b.points_xy)
            np.testing.assert_array_equal(a.vectors_xy, b.vectors_xy)
            exact_tensor(old._state.nodes.activity, new._state.nodes.activity)
            exact_tensor(old._last_hex, new._last_hex)
            exact_tensor(old._baseline_flow, new._baseline_flow)
            np.testing.assert_array_equal(old.baseline_activity, new.baseline_activity)
            if index:
                exact_tensor(old.last_decoded, new.last_decoded)
                np.testing.assert_array_equal(old.last_activity, new.last_activity)
                assert old.last_activity.dtype == new.last_activity.dtype
                for aggregation in ('mean', 'last', 'sum'):
                    first = old.field_from_last(aggregation)
                    second = new.field_from_last(aggregation)
                    np.testing.assert_array_equal(first.points_xy, second.points_xy)
                    np.testing.assert_array_equal(first.vectors_xy, second.vectors_xy)
                    arrays[f'{mode}_{temporal}_{index}_{aggregation}'] = second.vectors_xy
            else:
                assert old.last_decoded is new.last_decoded is None
                assert old.last_activity is new.last_activity is None
        assert old.stats['observations'] - old_count == new.stats['observations'] - new_count == len(frames)
        checks.append(dict(mode=mode, temporal=temporal, frames=len(frames), all_state_activity_fields_exact=True))
        print(mode, temporal, '12-frame state/output parity exact', flush=True)

failures = []
for phase in ('first', 'later'):
    for model in (old, new):
        model.mode = 'full'
        model.temporal_resampling = 'linear'
        model.reset()
    if phase == 'later':
        old.observe(frames[0]); new.observe(frames[0])
    before_stats = dict(new.stats)
    before_state = new._state
    decoder = new.decoder
    def invalid(activity):
        output = decoder(activity)
        return torch.full_like(output, float('nan'))
    new.decoder = invalid
    try:
        try:
            new.observe(frames[9])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid decoder output accepted')
    finally:
        new.decoder = decoder
    assert new.stats == before_stats and new._state is before_state
    if phase == 'first':
        assert new._frame_size is None and new._bounds is None
    clean = frames[0] if phase == 'first' else frames[1]
    a, b = old.observe(clean), new.observe(clean)
    np.testing.assert_array_equal(a.points_xy, b.points_xy)
    np.testing.assert_array_equal(a.vectors_xy, b.vectors_xy)
    exact_tensor(old._state.nodes.activity, new._state.nodes.activity)
    failures.append(dict(phase=phase, retry_exact=True, stats_unchanged=True))

np.savez_compressed(OUT / 'native_outputs.npz', **arrays)
report = dict(original_module_sha256=hashlib.sha256((OUT / 'before_flyvis_motion.py').read_bytes()).hexdigest(),
              current_module_sha256=hashlib.sha256((ROOT / 'tracking/flyvis_motion.py').read_bytes()).hexdigest(),
              native_device='cpu', cuda_initialized=torch.cuda.is_initialized(), checks=checks,
              injected_native_decoder_nan=failures, sequences=4, observations_per_estimator=48,
              field_comparisons=4*11*3, seconds=time.monotonic()-start,
              limits='Native numerical/error-retry regression checks only; no GPU, training, HOTA or scientific benefit claim')
assert not report['cuda_initialized']
(OUT / 'native_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
