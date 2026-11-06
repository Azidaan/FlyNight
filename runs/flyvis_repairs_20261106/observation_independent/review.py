"""Independent failure/retry review of observe publication boundaries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tracking.flyvis_motion import FlyVisMotionEstimator,TileLayout
torch.set_num_threads(2)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=ROOT/'tracking/flyvis_motion.py';source_hash=sha(source)
baseline=ROOT/'runs/flyvis_repairs_20261106/observation_transaction/flyvis_motion.py'
spec=importlib.util.spec_from_file_location('independent_original_observe',baseline)
old_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_module;spec.loader.exec_module(old_module)
old_cls=old_module.FlyVisMotionEstimator

def make(cls=FlyVisMotionEstimator,mode='full',dtype=torch.float64,resampling='linear',aggregation='mean',subtract=True):
    e=cls.__new__(cls);e.torch=torch;e.flyvis=SimpleNamespace(device=torch.device('cpu'))
    e.mode=mode;e.layout=TileLayout(1,2,.2);e.eye_width=e.eye_height=5;e.kernel_area=1;e.flow_gain=1.
    e.geometry_version='legacy_v1';e._local_centers_yx=np.array([[0,0],[-1,1],[1,-1]],np.float32)
    e.integration_steps=3;e.dt=.01;e.temporal_resampling=resampling;e.flow_aggregation=aggregation
    e.subtract_initial_baseline=subtract;e.stats=dict(observations=0,render_seconds=0.,network_seconds=0.,decoder_seconds=0.)
    e._timed=lambda function:(function(),.125);e.failure=None;e.reset()
    e.eye=lambda x:x.flatten(-2)[:,:,[6,12,18]].unsqueeze(2).to(dtype=dtype)
    def state(x):return SimpleNamespace(nodes=SimpleNamespace(activity=x))
    def initial(rendered):
        if e.failure=='initial_raise':raise RuntimeError('initial failure')
        n=5 if e.failure=='initial_wrong_nodes' else 4
        activity=torch.ones((rendered.shape[0],n),dtype=dtype)
        if e.failure=='initial_nan':activity[0,0]=float('nan')
        return state(activity)
    e._initial_state=initial
    def simulate(stimulus,dt,initial_state,as_states):
        if e.failure=='network_raise':raise RuntimeError('network failure')
        count=e.integration_steps
        if e.failure=='zero_steps':count=0
        if e.failure=='short_steps':count-=1
        if e.failure=='long_steps':count+=1
        values=[];previous=initial_state.nodes.activity
        for i in range(count):
            if e.failure=='custom_inplace':
                previous.add_(1.)
                raise RuntimeError('unsupported custom in-place backend')
            current=previous+stimulus[:,min(i,stimulus.shape[1]-1)].mean((1,2))[:,None]+(i+1)*.01
            if e.failure=='activity_nan':current=current*float('nan')
            values.append(state(current));previous=current
        return values
    e.network=SimpleNamespace(simulate=simulate,n_nodes=4)
    def decoder(activity):
        if e.failure=='decoder_raise':raise RuntimeError('decoder failure')
        b,t,_=activity.shape
        if e.failure=='decoder_wrong_time':t+=1
        channels=3 if e.failure=='decoder_wrong_channels' else 2
        value=torch.nan_to_num(activity).mean(-1)[:,:,None,None].expand(b,activity.shape[1],2,3).clone()
        if t!=activity.shape[1] or channels!=2:value=torch.ones((b,t,channels,3),dtype=dtype)
        if e.failure=='decoder_nan':value[0,0,0,0]=float('nan')
        if e.failure=='decoder_overflow':value=torch.full_like(value,1e300 if dtype==torch.float64 else torch.finfo(dtype).max)
        return value
    e.decoder=decoder
    points=e._points_and_vectors
    def produce(flow):
        if e.failure=='field_raise':raise RuntimeError('field construction failure')
        return points(flow)
    e._points_and_vectors=produce
    return e

NAMES=('_state','_last_hex','_baseline_flow','baseline_activity','last_activity','last_decoded','_frame_size','_bounds')
def snapshot(e):
    ids={name:id(getattr(e,name)) for name in NAMES}
    data={}
    for name in NAMES:
        item=getattr(e,name)
        if name=='_state' and item is not None:item=item.nodes.activity
        if isinstance(item,torch.Tensor):data[name]=item.detach().cpu().numpy().copy()
        elif isinstance(item,np.ndarray):data[name]=item.copy()
        else:data[name]=copy.deepcopy(item)
    return dict(ids=ids,data=data,stats=e.stats.copy())
def unchanged(e,before):
    after=snapshot(e)
    if after['ids']!=before['ids'] or after['stats']!=before['stats']:return False
    for name,a in after['data'].items():
        b=before['data'][name]
        if isinstance(a,np.ndarray):
            if not np.array_equal(a,b,equal_nan=True):return False
        elif a!=b:return False
    return True
def same(a,b):
    assert a.stats==b.stats
    assert torch.equal(a._state.nodes.activity,b._state.nodes.activity)
    assert torch.equal(a._last_hex,b._last_hex)
    assert torch.equal(a._baseline_flow,b._baseline_flow)
    assert np.array_equal(a.baseline_activity,b.baseline_activity)
    if a.last_activity is not None:
        assert a.last_activity.dtype==b.last_activity.dtype
        assert np.array_equal(a.last_activity,b.last_activity)
        for agg in ['mean','last','sum']:
            x=a.field_from_last(agg);y=b.field_from_last(agg)
            assert np.array_equal(x.points_xy,y.points_xy) and np.array_equal(x.vectors_xy,y.vectors_xy)

frames=[np.arange(80,dtype=np.uint8).reshape(8,10)+v for v in (0,7,13,21)]
first_faults=['initial_raise','initial_wrong_nodes','initial_nan','decoder_raise','decoder_wrong_time','decoder_wrong_channels','decoder_nan','field_raise']
later_faults=['network_raise','zero_steps','short_steps','long_steps','activity_nan','decoder_raise','decoder_wrong_time','decoder_wrong_channels','decoder_nan','decoder_overflow','field_raise']
cases=[]
with warnings.catch_warnings(),np.errstate(all='ignore'):
    warnings.simplefilter('ignore')
    for mode,dtype in itertools.product(['full','tiled'],[torch.float32,torch.float64]):
        for first,failures in [(True,first_faults),(False,later_faults)]:
            for failure in failures:
                current=make(mode=mode,dtype=dtype);control=make(mode=mode,dtype=dtype)
                if not first:
                    for frame in frames[:2]:current.observe(frame);control.observe(frame)
                before=snapshot(current);current.failure=failure
                try:current.observe(frames[0] if first else frames[2])
                except (ValueError,RuntimeError):pass
                else:raise AssertionError(('fault not rejected',first,failure,mode,dtype))
                assert unchanged(current,before),(first,failure,mode,dtype)
                current.failure=None
                field=current.observe(frames[0] if first else frames[2])
                reference=control.observe(frames[0] if first else frames[2])
                assert np.array_equal(field.vectors_xy,reference.vectors_xy);same(current,control)
                assert current.baseline_activity.dtype==np.dtype(str(dtype).split('.')[-1])
                if not first:assert current.last_activity.dtype==np.dtype(str(dtype).split('.')[-1])
                cases.append(dict(first=first,failure=failure,mode=mode,dtype=str(dtype),state_and_aliases_unchanged=True,retry_exact=True))

invalid_frames=[np.zeros((0,10),np.uint8),np.zeros((8,10,2),np.uint8),np.full((8,10),np.nan),np.zeros((8,10),np.float16)]
for first,frame in itertools.product([True,False],invalid_frames):
    e=make()
    if not first:
        e.observe(frames[0]);e.observe(frames[1])
    before=snapshot(e)
    try:e.observe(frame)
    except Exception:pass
    else:raise AssertionError('malformed input accepted')
    assert unchanged(e,before)
    cases.append(dict(first=first,malformed_shape=list(frame.shape),dtype=str(frame.dtype),state_and_aliases_unchanged=True))

valid=[]
for mode,dtype,resampling,aggregation,subtract in itertools.product(['full','tiled'],[torch.float32,torch.float64],['hold','linear'],['mean','last','sum'],[False,True]):
    current=make(mode=mode,dtype=dtype,resampling=resampling,aggregation=aggregation,subtract=subtract)
    original=make(old_cls,mode=mode,dtype=dtype,resampling=resampling,aggregation=aggregation,subtract=subtract)
    for frame in frames:
        a=current.observe(frame);b=original.observe(frame)
        assert np.array_equal(a.points_xy,b.points_xy) and np.array_equal(a.vectors_xy,b.vectors_xy)
        same(current,original)
    valid.append(dict(mode=mode,dtype=str(dtype),resampling=resampling,aggregation=aggregation,subtract=subtract,frames=len(frames),exact=True))

# Confirm that the repair changes the demonstrated failure behavior of the
# original; this is not a test that merely echoes the new implementation.
old=make(old_cls);old.observe(frames[0]);old.observe(frames[1]);old.failure='decoder_nan';before=snapshot(old)
try:old.observe(frames[2])
except ValueError:pass
old_failure_mutates=not unchanged(old,before);assert old_failure_mutates

# Explicitly retain the bounded claim: external custom in-place simulators
# mutate the old object itself and are not rolled back by wrapper staging.
e=make();e.observe(frames[0]);before=snapshot(e);e.failure='custom_inplace'
try:e.observe(frames[1])
except RuntimeError:pass
custom_inplace_not_rolled_back=not unchanged(e,before);assert custom_inplace_not_rolled_back
assert sha(source)==source_hash and not torch.cuda.is_initialized()
report=dict(passed=True,source_sha256=source_hash,baseline_sha256=sha(baseline),fault_cases=cases,
            valid_old_new_cases=valid,old_failure_mutates=True,float64_preserved=True,
            boundary=dict(custom_inplace_not_rolled_back=True,meaning='Wrapper publication is atomic for functional backend outputs. It is not a transaction for arbitrary backend mutation, RNG, weights or external side effects.'),
            cuda_initialized=False,script_sha256=sha(__file__))
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,fault_cases=len(cases),valid_old_new_cases=len(valid),valid_frames=sum(v['frames'] for v in valid),source_sha256=source_hash,boundary=report['boundary']),indent=2))
