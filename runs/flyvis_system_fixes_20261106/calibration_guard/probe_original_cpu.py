"""Frozen pre-audit estimator vs current stable output, both on CPU."""
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
from tracking.flyvis_adapter import activate_flyvis
from scripts.flyvis_motion_calibration import textured_frame, shifted, SHIFTS

snapshot = ROOT / 'runs/flyvis_system_fixes_20261106/motion/original/flyvis_motion.py'
spec = importlib.util.spec_from_file_location('frozen_pre_audit_motion', snapshot)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
old.activate_flyvis = lambda: activate_flyvis(device='cpu')
start = time.monotonic()
model = old.FlyVisMotionEstimator(mode='full', source_fps=5., dt=.02)
assert next(model.network.parameters()).device.type == 'cpu'
base = textured_frame()
records = []
maximum = 0.
arrays = {}
with np.load(OUT / 'native_outputs.npz') as actual:
    for index, (dx, dy) in enumerate(SHIFTS):
        model.reset()
        model.observe(base)
        model.observe(shifted(base, dx, dy))
        for aggregation in ('mean', 'last', 'sum'):
            previous = model.field_from_last(aggregation).vectors_xy
            current = actual[f'full_{index}_{aggregation}_stable']
            maximum = max(maximum, float(np.max(np.abs(previous - current))))
            np.testing.assert_array_equal(previous, current)
            arrays[f'full_{index}_{aggregation}'] = previous
        predicted = np.median(model.field_from_last('mean').vectors_xy, axis=0)
        records.append(dict(true_xy=[dx, dy], predicted_xy=predicted.astype(float).tolist()))
        print('original shift', dx, dy, 'exact stable full estimator vectors', flush=True)
target = np.asarray([r['true_xy'] for r in records])
predicted = np.asarray([r['predicted_xy'] for r in records])
np.savez_compressed(OUT / 'original_native_outputs.npz', **arrays)
report = dict(original_source_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              only_override='activate_flyvis(device=cpu); all original estimator methods unchanged',
              original_cpu_vs_current_stable_vectors_exact=True, max_vector_difference=maximum,
              original_cpu_rmse=float(np.sqrt(np.mean((predicted - target) ** 2))),
              records=records, cuda_initialized=torch.cuda.is_initialized(), seconds=time.monotonic() - start,
              interpretation='The saved historical calibration differs slightly from BOTH frozen original CPU and current CPU. This probe does not establish the cause of that historical/runtime difference. Pixel-centers change itself leaves vectors and global median calibration exactly unchanged.')
(OUT / 'original_native_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
