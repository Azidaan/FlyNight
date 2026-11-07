"""Bounded temporal boundary checks; intercept large work before allocation."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,copy
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
import tracking.b1_timestamped as timed
from tracking.b1_tracker_runtime import ResidualEngine
from tracking.b1_learning import ReceptorHistory
OUT=Path(__file__).resolve().parent
assert not (OUT/'time_report.json').exists()
x=torch.arange(20,dtype=torch.float32).reshape(5,4)/100
checks=[]

def snapshot(eye):
    return dict(eye),{k:v.clone() for k,v in eye.items() if torch.is_tensor(v)}
def same(eye,saved):
    refs,values=saved
    return eye.keys()==refs.keys() and all(eye[k] is refs[k] for k in refs) and all(torch.equal(eye[k],v) for k,v in values.items())
def engine(fps=30.):
    bb,h=timed.make_timestamped_modules('R1',device='cpu')
    return timed.TimestampedResidualEngine(bb,h,timed.timestamped_checkpoint(bb,h,cursor=0),source_fps=fps)

control=engine();expected=control.start([dict(frame=i,hex=row) for i,row in enumerate(x[:4])]);control.advance_at(expected,x[4],4)
for value in [4,np.int64(4),np.uint64(4),4.,np.float32(4),True,np.bool_(True),np.nan,np.inf,-1,3,2,None]:
    obj=engine();eye=obj.start([dict(frame=i,hex=row) for i,row in enumerate(x[:4])]);old=snapshot(eye);error=None
    try:obj.advance_at(eye,x[4],value)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    if error:assert same(eye,old)
    else:assert eye['last_frame']==4 and torch.equal(eye['state'],expected['state'])
    checks.append(dict(kind='live_frame',value=repr(value),type=type(value).__name__,error=error,unchanged_on_error=bool(error and same(eye,old)),valid_exact=not bool(error)))

for fps in [np.int64(30),np.float32(30),np.nan,np.inf,-np.inf,0.,-1.,True,'30']:
    error=None
    try:obj=engine(fps)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    checks.append(dict(kind='source_fps',value=repr(fps),type=type(fps).__name__,error=error,stored=None if error else obj.source_fps))

obj=engine();bb=obj.bb
history=torch.stack([x[0],x[0],x[1]])[None]
_,repeat=bb.sequence(history,[1,1,2],30)
_,single=bb.sequence(history[:,[0,2]],[1,2],30)
assert torch.equal(repeat[-1],single[-1])
for frames in ([2,1,3],[1,1,2]):
    hx=history.clone()
    if frames==[1,1,2]:hx[:,1].add_(.1)
    try:bb.sequence(hx,frames,30)
    except ValueError as exc:checks.append(dict(kind='history_rejected',frames=frames,error=str(exc)))
    else:raise AssertionError('Backwards or inconsistent duplicate accepted')
checks.append(dict(kind='identical_duplicate',zero_elapsed_exact=True))
offset=10**400
huge=engine();eye=huge.start([dict(frame=offset+i,hex=row) for i,row in enumerate(x[:4])]);huge.advance_at(eye,x[4],offset+4)
assert torch.equal(eye['state'],expected['state'])
checks.append(dict(kind='huge_absolute_frame_small_gap',digits=401,valid_exact=True))

class WorkIntercept(RuntimeError):pass
obj=engine();eye=obj.start([dict(frame=i,hex=row) for i,row in enumerate(x[:4])]);old=snapshot(eye)
calls=[];raw_step=obj.bb.backend.step
def bounded_step(*args):
    calls.append(1)
    if len(calls)==3:raise WorkIntercept('Stopped after 2 actual steps; requested gap is 10**12')
    return raw_step(*args)
with patch.object(obj.bb.backend,'step',bounded_step):
    try:obj.advance_at(eye,x[4],3+10**12)
    except WorkIntercept:pass
    else:raise AssertionError('Expected interception')
assert same(eye,old)
checks.append(dict(kind='R1_huge_gap_resource_boundary',requested_steps=10**12,completed_actual_steps=2,
    no_builtin_work_limit=True,old_eye_preserved_on_intercept=True,limit='Finite loop demand inferred from source and intercepted; no hang executed.'))

class Net:
    stimulus=SimpleNamespace(n_nodes=2,input_index=np.array([[0,1]]))
    def forward(self,values,dt,state,as_states):
        self.last_dt=dt
        return [SimpleNamespace(nodes=SimpleNamespace(activity=values[:,-1]))]
net=Net();raw=SimpleNamespace(dev=torch.device('cpu'),net=net,index=torch.tensor([[0,1]]))
bb=timed.TimestampedBackbone(raw,'D');state=SimpleNamespace(nodes=SimpleNamespace(activity=torch.zeros(1,2)))
for gap,fps in [(10**12,30.),(1,1e-320),(1,np.nextafter(0.,1.)),(1,1e308),(10**400,30.)]:
    allocations=[];error=None
    original_arange=torch.arange
    def guarded_arange(*args,**kwargs):
        allocations.append([str(a) for a in args])
        if len(args)>1 and args[1]-args[0]>10000:raise WorkIntercept('Large arange intercepted before allocation')
        return original_arange(*args,**kwargs)
    with patch.object(torch,'arange',guarded_arange):
        try:bb.step(state,torch.zeros(1,2),torch.ones(1,2),previous_frame=0,current_frame=gap,source_fps=fps)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    checks.append(dict(kind='C_D_duration_boundary',gap=str(gap),fps=repr(fps),allocation_requests=allocations,error=error,
        dt=None if error else net.last_dt,original_state_finite=bool(torch.isfinite(state.nodes.activity).all())))

initial_flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
try:
    torch.use_deterministic_algorithms(False,warn_only=True)
    for arm in ('R1','B'):
        before=torch.random.get_rng_state().clone()
        def failed_head():
            torch.rand(7,device='cpu')
            raise RuntimeError('Constructor fails after consuming CPU random values')
        with patch.object(timed,'TemporalResidualAssociation',failed_head):
            try:timed.make_timestamped_modules(arm,device='cpu',neighbors=np.tile(np.arange(4)[:,None],(1,7)))
            except RuntimeError:pass
            else:raise AssertionError('Expected constructor failure')
        assert torch.equal(before,torch.random.get_rng_state())
        assert not torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
        checks.append(dict(kind='failed_factory',arm=arm,RNG_preserved=True,deterministic_flags_preserved=True))
    for defect in ('head_shape','python_nan_optimizer'):
        bb,h=timed.make_timestamped_modules('R1',device='cpu');opt=torch.optim.Adam(h.parameters())
        ck=timed.timestamped_checkpoint(bb,h,cursor=1,optimizer=opt)
        saved={k:v.clone() for k,v in h.state_dict().items()}
        if defect=='head_shape':ck['head']['output.weight']=torch.zeros(1,99)
        else:ck['optimizer']['param_groups'][0]['lr']=float('nan')
        before=torch.random.get_rng_state().clone()
        try:timed.restore_timestamped_checkpoint(ck,bb,h,optimizer=opt)
        except ValueError:pass
        else:raise AssertionError('Expected restore rejection')
        assert all(torch.equal(saved[k],v) for k,v in h.state_dict().items())
        assert torch.equal(before,torch.random.get_rng_state())
        assert not torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
        checks.append(dict(kind='failed_restore',defect=defect,model_RNG_and_flags_preserved=True))
finally:torch.use_deterministic_algorithms(*initial_flags[:1],warn_only=initial_flags[1])

# Historical semantics intentionally ignore frame metadata and advance once per
# supplied observation; preserving this is not interpreted as timestamp support.
old=ResidualEngine.__new__(ResidualEngine);old.arm='R1';old.bb=ReceptorHistory('cpu')
a=old.start([dict(frame=i,hex=row) for i,row in enumerate(x[:4])])
b=old.start([dict(frame=99-i,hex=row) for i,row in enumerate(x[:4])])
assert torch.equal(a['state'],b['state'])
checks.append(dict(kind='historical_boundary',frame_metadata_unused=True,timing_unchanged=True,
    limit='This preserves historical fixed-step semantics; it is not a new timing correction.'))
assert not torch.cuda.is_initialized()
report=dict(checks=checks,cuda_initialized=False,production_sources_edited=False,
    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['tracking/b1_timestamped.py','tracking/b1_tracker_runtime.py']},
    boundary='Synthetic CPU checks only. Large resource requests were intercepted before allocation/long loops. No full training, native FlyVis evaluation or old-time policy migration.')
(OUT/'time_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
