"""Existing targeted tests, CPU only, with explicit FlyVis device aliases."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys, importlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
from tracking.flyvis_adapter import activate_flyvis
fv=activate_flyvis()
for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
torch.set_default_device('cpu')
import pytest
names=['flyvis_tracker','flyvis_adapter','flyvis_activity_readout','foveated_eyes','retina','flyvis_temporal_residual','b1_eye_layout','b1_fast_stimulus','b1_index_copy_stimulus','b1_cached_eye','b1_learning','b1_real_inputs','b1_tracker_runtime','b2_feature_temporal','b2_causal_detector','b2_tracker_runtime','b3_object_memory','query_assignment_write','query_bounded_write','recovery_detection_support','history_pair_runtime']
args=[str(ROOT/'tests'/('test_'+n+'.py')) for n in names]
raise SystemExit(pytest.main(args+['-q','--tb=short','-p','no:cacheprovider','--basetemp',str(Path(__file__).parent/'pytest_tmp')]))
