"""Отдельный причинный контроль: константы ×0,5 с прежним ограничением шага."""
import os
os.environ['Y1_EXECUTION_DEVICE']='cpu'
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tracking.flyvis_motion import FlyVisMotionEstimator
from scripts import flyvis_y1_latency_fp64_20261104 as W
OUT=ROOT/'runs/flyvis_tau_half_20261104'
original_init=FlyVisMotionEstimator.__init__

def patched(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    before={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
    with torch.no_grad():self.network.nodes_time_const.mul_(.5)
    after=self.network.state_dict()
    changed=[k for k in before if not torch.equal(before[k],after[k].cpu())]
    assert changed==['nodes_time_const'],changed
    raw=before['nodes_time_const'].numpy();new=after['nodes_time_const'].detach().cpu().numpy()
    assert np.array_equal(new,raw*.5)
    original_effective=np.maximum(raw,self.dt);new_effective=np.maximum(new,self.dt)
    (OUT/'tau_intervention.json').write_text(json.dumps(dict(multiplier=.5,dt=self.dt,parameter_count=len(raw),
       original_raw=raw.tolist(),new_raw=new.tolist(),original_effective=original_effective.tolist(),
       new_effective=new_effective.tolist(),clamped_count=int((new<=self.dt).sum()),
       changed_parameter_keys=changed,other_parameters_exact=True),indent=2)+'\n',encoding='utf-8')

FlyVisMotionEstimator.__init__=patched
W.OUT=OUT
if __name__=='__main__':W.infer()
