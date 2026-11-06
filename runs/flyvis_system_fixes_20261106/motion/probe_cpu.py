"""Fixed CPU regression: exact old geometry path and finite short warmup paths.

No training or quality claim. Shared stock model/data; original timer bypassed
only to prevent its unrelated unconditional CUDA synchronization.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys, importlib.util, hashlib, json
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator
new=FlyVisMotionEstimator(source_fps=30,flow_gain=1,fade_in_seconds=.1,
                         temporal_resampling='linear',geometry_version='legacy_v1',device='cpu')
spec=importlib.util.spec_from_file_location('old_flyvis_motion',OUT/'original/flyvis_motion.py')
oldmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oldmod;spec.loader.exec_module(oldmod)
old=oldmod.FlyVisMotionEstimator(source_fps=30,flow_gain=1,fade_in_seconds=.1,temporal_resampling='linear')
old._synchronize=lambda:None
frames=np.random.default_rng(709).integers(0,256,(3,96,128),dtype=np.uint8)
def run(model):
    model.reset();fields=[model.observe(frame) for frame in frames]
    return np.stack([f.points_xy for f in fields]),np.stack([f.vectors_xy for f in fields])
oldpoints,oldflow=run(old);points,flow=run(new)
assert np.array_equal(points,oldpoints) and np.array_equal(flow,oldflow)
new.geometry_version='resize_v2';newpoints,newflow=run(new)
assert np.array_equal(newflow[:,:,1],oldflow[:,:,1])
assert np.allclose(newflow[:,:,0],oldflow[:,:,0]*128/96,rtol=2e-6,atol=1e-7)
warm=[]
for fade in (0.,.005):
    new.fade_in_seconds=fade;p,f=run(new)
    warm.append({'fade':fade,'finite':bool(np.isfinite(f).all()),'first_frame_zero':bool(np.all(f[0]==0))})
    assert warm[-1]['finite'] and warm[-1]['first_frame_zero']
assert not torch.cuda.is_initialized()
np.savez_compressed(OUT/'probe_outputs.npz',frames=frames,legacy_points=oldpoints,legacy_flow=oldflow,
                    new_points=newpoints,new_flow=newflow)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'passed':True,'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),
   'legacy_vs_snapshot_points_exact':np.array_equal(points,oldpoints),
   'legacy_vs_snapshot_vectors_exact':np.array_equal(flow,oldflow),
   'v2_vertical_exact':np.array_equal(newflow[:,:,1],oldflow[:,:,1]),
   'v2_horizontal_ratio_width_over_height':128/96,'warmups':warm,
   'scope':'one fixed synthetic input; no HOTA or optimization; old timer bypassed only',
   'sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'probe_outputs.npz',
       OUT/'original/flyvis_motion.py',ROOT/'tracking/flyvis_motion.py',
       ROOT/'tracking/flyvis_adapter.py',ROOT/'tracking/flyvis_checkpoint.py']}}
(OUT/'probe_report.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
