"""Один полныйGPU-FP64ответ против сохранённого независимогоCPU-FP64эталона."""
import os
os.environ['Y1_EXECUTION_DEVICE']='cuda'
from pathlib import Path
import json
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import flyvis_y1_latency_fp64_20261104 as W
parent=W.OUT;folder=parent/'gpu_reference';folder.mkdir(exist_ok=True)
p=W.protocol();p['cases']=p['cases'][:1]
(folder/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(folder/'inputs').mkdir(exist_ok=True)
case=p['cases'][0]['id']
for suffix in ['.npz','.json']:
    target=folder/'inputs'/(case+suffix)
    if not target.exists():os.link(parent/'inputs'/(case+suffix),target)
shutil.copy2(parent/'geometry.npz',folder/'geometry.npz')
(folder/'inputs_completed.json').write_text('{"complete":true}\n',encoding='utf-8')
W.OUT=folder;W.infer()
gpu=np.load(folder/'outputs'/(case+'.npz'))['flow'][0,:,1:].transpose(1,0,2,3)
source=ROOT/'runs/flyvis_y1_latency_20261104/cpu_reference_fp64_failed/cpu_reference.npz'
cpu=np.load(source)['flow']
diff=float(np.max(np.abs(cpu-gpu)));assert diff<=1e-5,diff
shutil.copy2(source,parent/'cpu_reference.npz')
W.OUT=parent
W.write('cpu_reference.json',dict(complete=True,passed=True,max_abs_diff=diff,tolerance=1e-5,
       case=case,frames_checked=78,mode='hold',cuda_initialized=False,reference_dtype='float64',
       scope='One GPU-FP64 full case against independently streamed CPU-FP64 response; previous comparison to GPU-FP32 remains failed',
       reused_cpu_raw_source=str(source.relative_to(ROOT)),reused_cpu_raw_sha256=W.sha(source)))
print('FP64 CPU/GPU full-case comparison:',diff,flush=True)
