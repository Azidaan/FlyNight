"""Service-only CPU recovery: NVML availability may disagree with hidden CUDA devices.

Preserves frozen scientific runner and failed first benchmark artifacts.
"""
import os
import runpy
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
os.environ['TIMEZONE']='Europe/Moscow'
import torch
torch.cuda.is_available=lambda:False
torch.set_default_device('cpu')
assert torch.empty(1).device.type=='cpu' and not torch.cuda.is_initialized()
runpy.run_path(str(Path(__file__).with_name('z13_target_center_probe.py')),run_name='__main__')
