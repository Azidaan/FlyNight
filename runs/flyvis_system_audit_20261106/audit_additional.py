"""Additional engineering checks, fixed before execution.

Criteria: existing stock keys finite; no optimizer updates; patched dynamics
matches original for same state/input; gradient exists for each FREE group.
Missing-payload probe must record whether official loader rejects the input.
Only scratch objects are modified; no on-disk model/implementation changes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,hashlib,importlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
from tracking.flyvis_adapter import activate_flyvis
fv=activate_flyvis()
for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
torch.set_default_device('cpu')
from flyvis.utils.chkpt_utils import recover_network,recover_decoder
from tracking.flyvis_motion import FlyVisMotionEstimator
from tracking.z13_readout import FlyBackbone,FREE
r={}
for name,fn in [('network',lambda m:recover_network(m,{})),('decoder',lambda m:recover_decoder({'flow':m},{'decoder':{}}))]:
    m=torch.nn.Linear(2,2);before={k:v.clone() for k,v in m.state_dict().items()}
    try:
        fn(m);r[name+'_missing_payload']={'raised':False,'initialization_retained':all(torch.equal(v,before[k]) for k,v in m.state_dict().items())}
    except Exception as e:r[name+'_missing_payload']={'raised':True,'error':repr(e)}
model=FlyVisMotionEstimator(source_fps=30,flow_gain=1,fade_in_seconds=.1)
net=model.network;original_velocity=net.dynamics.write_state_velocity
bb=FlyBackbone(model,trainable=True)
before={k:v.clone().detach() for k,v in net.state_dict().items()}
torch.manual_seed(19);x=torch.rand(1,1,721);y=torch.rand(1,721)
state=bb.init(x)
_,patched=bb.step(state,x[:,0],y)
net.dynamics.write_state_velocity=original_velocity
_,original=bb.step(state,x[:,0],y)
r['constant_dt_patch']={'max_abs_diff':float((original-patched).abs().max().detach()),'exact':torch.equal(original,patched)}
original.square().mean().backward()
r['gradient']={k:{'present':v.grad is not None,'finite':bool(torch.isfinite(v.grad).all()) if v.grad is not None else False,'norm':float(v.grad.norm()) if v.grad is not None else None} for k,v in net.named_parameters() if k in FREE}
r['weights_unchanged']=all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
r['network_devices']=sorted({str(v.device) for v in net.parameters()})
p=ROOT/'runs/continuation_20261028/b2_sparse_training_pilot/checkpoints/final.pt'
z=torch.load(p,map_location='cpu',weights_only=False)
s=z['models']['D']['backend'];net.load_state_dict(s)
loaded={k:v.clone().detach() for k,v in net.state_dict().items()};net.clamp()
r['b2_D_projection']={'checkpoint':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':{k:float((v-loaded[k]).abs().max()) for k,v in net.state_dict().items()},'note':'Current saved checkpoint constraints only; not a reproduction of YOLO dev losses.'}
stock=[]
for p in sorted((ROOT/'runs/flyvis/results/flow/0000').glob('*/best_chkpt')):
    z=torch.load(p,map_location='cpu',weights_only=False)
    stock.append({'path':str(p.relative_to(ROOT)),'all_tensors_finite':all(bool(torch.isfinite(v).all()) for s in [z['network'],z['decoder']['flow']] for v in s.values())})
r['all_stock_finite']=stock
r['cuda_initialized']=torch.cuda.is_initialized()
r['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'additional_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in r.items() if k!='all_stock_finite'},indent=2))
