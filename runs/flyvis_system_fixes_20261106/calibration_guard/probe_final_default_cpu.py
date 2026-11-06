"""Final ordinary defaults: native full/tiled fields equal frozen old converter."""
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
from tracking.flyvis_motion import FlyVisMotionEstimator, GEOMETRY_VERSION

snapshot = ROOT / 'runs/flyvis_system_fixes_20261106/motion/original/flyvis_motion.py'
spec = importlib.util.spec_from_file_location('frozen_default_converter', snapshot)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
start = time.monotonic()
model = FlyVisMotionEstimator(device='cpu')
assert model.geometry_version == GEOMETRY_VERSION == 'legacy_v1'
frames = [np.random.default_rng(i + 414).integers(0, 256, (96, 128, 3), dtype=np.uint8) for i in range(3)]
arrays = {}
cases = []
for mode in ('full', 'tiled'):
    model.mode = mode
    model.reset()
    for index, frame in enumerate(frames):
        field = model.observe(frame)
        if index == 0:
            raw = torch.zeros((len(model._bounds), 2, len(model._local_centers_yx)), device='cpu')
            expected = old.FlyVisMotionEstimator._points_and_vectors(model, raw)
            np.testing.assert_array_equal(field.points_xy, expected.points_xy)
            np.testing.assert_array_equal(field.vectors_xy, expected.vectors_xy)
            continue
        for aggregation in ('mean', 'last', 'sum'):
            decoded = model.last_decoded
            raw = decoded.mean(1) if aggregation == 'mean' else decoded.sum(1) if aggregation == 'sum' else decoded[:, -1]
            actual = model.field_from_last(aggregation)
            expected = old.FlyVisMotionEstimator._points_and_vectors(model, raw)
            np.testing.assert_array_equal(actual.points_xy, expected.points_xy)
            np.testing.assert_array_equal(actual.vectors_xy, expected.vectors_xy)
            arrays[f'{mode}_{index}_{aggregation}_points'] = actual.points_xy
            arrays[f'{mode}_{index}_{aggregation}_vectors'] = actual.vectors_xy
            cases.append(dict(mode=mode, frame=index, aggregation=aggregation, points_and_vectors_exact=True))
np.savez_compressed(OUT / 'final_default_outputs.npz', **arrays)
report = dict(default_geometry=GEOMETRY_VERSION, native_device=str(next(model.network.parameters()).device),
              cuda_initialized=torch.cuda.is_initialized(), cases=cases,
              current_module_sha256=hashlib.sha256((ROOT / 'tracking/flyvis_motion.py').read_bytes()).hexdigest(),
              snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), seconds=time.monotonic() - start,
              interpretation='Ordinary geometry defaults restored to exact historical coordinates and vectors. New coordinate/axis transforms are explicit opt-in; state/input/checkpoint repairs remain active.')
(OUT / 'final_default_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
