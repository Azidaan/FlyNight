"""Native CPU calibration parity; no fitted gain selection or GPU execution."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import json
import hashlib
import importlib.util
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator, GEOMETRY_VERSION, SYNTHETIC_FLOW_GAIN
from scripts.flyvis_motion_calibration import textured_frame, shifted, SHIFTS

snapshot = ROOT / 'runs/flyvis_system_fixes_20261106/motion/original/flyvis_motion.py'
spec = importlib.util.spec_from_file_location('original_motion_conversion', snapshot)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
arrays = {}
results = []
began = time.monotonic()

def compare(model, prefix):
    for aggregation in ('mean', 'last', 'sum'):
        decoded = model.last_decoded
        raw = decoded.mean(1) if aggregation == 'mean' else decoded.sum(1) if aggregation == 'sum' else decoded[:, -1]
        stable = model.field_from_last(aggregation)
        historical = old.FlyVisMotionEstimator._points_and_vectors(model, raw)
        np.testing.assert_array_equal(stable.vectors_xy, historical.vectors_xy)
        arrays[prefix + '_' + aggregation + '_stable'] = stable.vectors_xy
        arrays[prefix + '_' + aggregation + '_historical'] = historical.vectors_xy
    return np.median(model.field_from_last('mean').vectors_xy, axis=0)

base = textured_frame()
model = FlyVisMotionEstimator(mode='full', source_fps=5., dt=.02, device='cpu')
assert model.geometry_version == GEOMETRY_VERSION == 'pixel_centers_v2'
assert model.flow_gain == SYNTHETIC_FLOW_GAIN
assert next(model.network.parameters()).device.type == 'cpu'
for index, (dx, dy) in enumerate(SHIFTS):
    model.reset()
    model.observe(base)
    model.observe(shifted(base, dx, dy))
    predicted = compare(model, 'full_' + str(index))
    results.append(dict(true_xy=[dx, dy], predicted_xy=predicted.astype(float).tolist()))
    print('full shift', dx, dy, 'exact legacy vectors', flush=True)

saved_path = ROOT / 'runs/flyvis/motion_calibration.json'
saved = json.loads(saved_path.read_text(encoding='utf-8'))
selected = [r for r in saved['records'] if r['temporal_resampling'] == 'hold' and r['aggregation'] == 'mean']
reference = np.asarray([r['predicted_xy'] for r in selected]) * SYNTHETIC_FLOW_GAIN
predicted = np.asarray([r['predicted_xy'] for r in results])
target = np.asarray([r['true_xy'] for r in results])
same_source_rmse = float(np.sqrt(np.mean((reference - target) ** 2)))
current_rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
research = np.asarray([r['predicted_xy'] for r in selected]) * [1280 / 720, 1.]
refit = float(np.sum(research * target) / np.sum(research * research))
counter = dict(old_rmse=same_source_rmse,
               resize_same_gain_rmse=float(np.sqrt(np.mean((research * SYNTHETIC_FLOW_GAIN - target) ** 2))),
               resize_refitted_gain_diagnostic_only=refit,
               resize_refitted_rmse=float(np.sqrt(np.mean((research * refit - target) ** 2))))

tiled = FlyVisMotionEstimator(mode='tiled', source_fps=30., dt=.02, device='cpu')
small = np.random.default_rng(9234).integers(0, 256, (96, 128, 3), dtype=np.uint8)
tiled.observe(small)
tiled.observe(np.roll(small, 2, axis=1))
compare(tiled, 'tiled')
np.savez_compressed(OUT / 'native_outputs.npz', **arrays)
report = dict(default_geometry=GEOMETRY_VERSION, default_gain=SYNTHETIC_FLOW_GAIN,
              source_calibration=str(saved_path.relative_to(ROOT)),
              source_calibration_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
              original_conversion_snapshot=str(snapshot.relative_to(ROOT)),
              original_conversion_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              new_module_sha256=hashlib.sha256((ROOT / 'tracking/flyvis_motion.py').read_bytes()).hexdigest(),
              native_device=str(next(model.network.parameters()).device), cuda_initialized=torch.cuda.is_initialized(),
              full_shifts=results, full_and_tiled_vectors_exactly_equal_original_converter_all_aggregations=True,
              native_rmse=current_rmse, native_vs_saved_calibrated_predictions_max_abs=float(np.abs(predicted - reference).max()),
              saved_calibration_counterevidence=counter, seconds=time.monotonic() - began,
              limits='CPU original-texture diagnostic, not HOTA or a new calibration; ROI point-selection effect requires separate check. Refitted resize gain is recorded as counterevidence only and is not used.')
(OUT / 'native_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
