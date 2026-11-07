"""Отдельная потоковая сверка одного полного случая ускоренной FlyVis на CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import hashlib
import importlib
import json
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_tau_half_20261104'
sys.path.insert(0, str(ROOT))
from tracking.flyvis_adapter import activate_flyvis
flyvis = activate_flyvis()
for module in (flyvis, importlib.import_module('flyvis.network.initialization'),
               importlib.import_module('flyvis.task.decoder')):
    module.device = torch.device('cpu')
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator
original_init = FlyVisMotionEstimator.__init__


def init_modified(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.network.double()
    self.decoder.double()
    self.eye.conv.double()
    with torch.no_grad():
        self.network.nodes_time_const.mul_(0.5)


def render64(self, frame):
    gray = np.asarray(frame)
    assert gray.ndim == 2 and np.isfinite(gray).all()
    height, width = gray.shape
    self._prepare_layout(width, height)
    arrays = [cv2.resize(gray[y1:y2, x1:x2], (self.eye_width, self.eye_height),
                         interpolation=cv2.INTER_AREA).astype(np.float64) / 255
              for x1, y1, x2, y2 in self._bounds]
    return self.eye(torch.from_numpy(np.stack(arrays))[:, None].to('cpu'))


FlyVisMotionEstimator.__init__ = init_modified
FlyVisMotionEstimator._render = render64
FlyVisMotionEstimator._synchronize = lambda self: None
protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
case = protocol['cases'][0]
inputs = np.load(OUT / 'inputs' / (case['id'] + '.npz'))
output_path = OUT / 'outputs' / (case['id'] + '.npz')
output = np.load(output_path)
model = FlyVisMotionEstimator(mode='full', source_fps=30, flow_gain=1,
                              temporal_resampling='hold', subtract_initial_baseline=False)
assert str(model.flyvis.device) == 'cpu' and not torch.cuda.is_initialized()
values, differences = [], []
for index, image in enumerate(inputs['frames']):
    model.observe(image)
    if index:
        decoded = model.last_decoded.cpu().numpy()[0]
        both = np.stack([decoded.mean(0), decoded[-1]], axis=0).transpose(0, 2, 1)
        both *= np.array([512 / 169, -512 / 169], np.float64)
        differences.append(float(np.max(np.abs(both - output['flow'][0, :, index]))))
        values.append(both)
np.savez_compressed(OUT / 'streaming_reference.npz', flow=np.stack(values))
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = dict(complete=True, passed=max(differences) <= 1e-5,
              max_abs_diff=max(differences), tolerance=1e-5, case=case['id'],
              frames_checked=len(differences), mode='hold', cuda_initialized=torch.cuda.is_initialized(),
              output_sha256=digest(output_path), source_sha256=digest(Path(__file__)),
              scope='One complete case via CPU streaming API; not independent replay of all 64 cases')
(OUT / 'streaming_reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
assert result['passed'] and not result['cuda_initialized'], result
