"""Reproduce import-side CUDA masking and verify the service restoration."""
from pathlib import Path
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2);torch.set_default_device('cpu')
before=os.environ['CUDA_VISIBLE_DEVICES']
from scripts import z13_assignment_effect_20261024
after_import=os.environ['CUDA_VISIBLE_DEVICES']
assert before=='0' and after_import==''
os.environ['CUDA_VISIBLE_DEVICES']='0'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
x=torch.ones(3,device='cuda:0');assert float(x.sum())==3.
from scripts.z13_candidate_reserve_20261021 import write,now
write(ROOT/'runs/z13_region_assignment_20261025/device_service_check.json',dict(at=now().isoformat(),before_import=before,after_cpu_helper_import=after_import,after_restore=os.environ['CUDA_VISIBLE_DEVICES'],device_count=torch.cuda.device_count(),tensor_device=str(x.device),tensor_sum=float(x.sum()),passed=True))
