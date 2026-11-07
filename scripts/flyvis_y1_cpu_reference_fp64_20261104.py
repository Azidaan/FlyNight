"""Более точный CPU-эталон той же модели/входов; исходный допуск1e-5 сохраняется."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import importlib
import sys
import json
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tracking.flyvis_adapter import activate_flyvis
flyvis = activate_flyvis()
for module in [flyvis, importlib.import_module('flyvis.network.initialization'), importlib.import_module('flyvis.task.decoder')]:
    module.device = torch.device('cpu')
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)
from tracking.flyvis_motion import FlyVisMotionEstimator
original_init = FlyVisMotionEstimator.__init__


def init64(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.network.double();self.decoder.double();self.eye.conv.double()


def render64(self, frame):
    gray = np.asarray(frame)
    assert gray.ndim == 2 and np.isfinite(gray).all()
    height, width = gray.shape;self._prepare_layout(width, height)
    arrays = [cv2.resize(gray[y1:y2, x1:x2], (self.eye_width, self.eye_height), interpolation=cv2.INTER_AREA).astype(np.float64) / 255
              for x1, y1, x2, y2 in self._bounds]
    return self.eye(torch.from_numpy(np.stack(arrays))[:, None].to('cpu'))


FlyVisMotionEstimator.__init__ = init64
FlyVisMotionEstimator._render = render64
FlyVisMotionEstimator._synchronize = lambda self: None  # CPU-таймеру не требуется CUDA.
from scripts.flyvis_y1_latency_20261104 import reference, OUT, write, sha
assert not torch.cuda.is_initialized()
reference()
result = json.loads((OUT / 'cpu_reference.json').read_text())
assert result['passed'] and not result['cuda_initialized']
result.update(reference_dtype='float64', source_sha256=sha(Path(__file__)),
              original_fp32_failure='cpu_reference_fp32_failed/cpu_reference.json',
              timing_only_fix='CPU _synchronize does not call CUDA',
              scope='One full fixed hold case on CPU in higher precision, including image preparation and BoxEye; same checkpoint values, equations, dt and raw output tolerance1e-5')
write('cpu_reference.json', result)
