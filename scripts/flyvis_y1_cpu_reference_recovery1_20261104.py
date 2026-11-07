"""Явное CPU во всех трёх модулях FlyVis до восстановления штатных весов."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tracking.flyvis_adapter import activate_flyvis
flyvis = activate_flyvis()
modules = [flyvis, importlib.import_module('flyvis.network.initialization'), importlib.import_module('flyvis.task.decoder')]
for module in modules:module.device = torch.device('cpu')
torch.set_default_device('cpu')
assert not torch.cuda.is_initialized()
from scripts.flyvis_y1_latency_20261104 import reference, write, sha
write('cpu_loader_recovery1.json', dict(device='cpu', modules=[m.__name__ for m in modules],
      cuda_initialized=False, source_sha256=sha(Path(__file__)),
      source='Existing three-module explicit CPU fix, without library or checkpoint changes'))
reference()
