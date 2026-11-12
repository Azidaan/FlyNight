"""Bounded unseen-reference parity and held C34 passes; parent GPU queue only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
    with (OUT/n).open('xb') as f:f.write((json.dumps(x,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
p=read(OUT/'protocol.json');assert read(OUT/'core_GO.json')['approved'] is True
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
import numpy as np,torch
from tracking.gpu_lock import GpuLock
from tracking.flyvis_adapter import activate_flyvis
from scripts.z13_target_model import build_arm
from tracking.reference_template_history import terminal_reference_history
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
norm=read(ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json');mu,sd=[np.array(norm[n])[:34] for n in ['mean','sd']];deadline=datetime.fromisoformat(p['deadline']);start=time.monotonic();unique={}
for c in p['cases']:
    ref=c['endpoints'][0];unique.setdefault(ref['key'],ref)
records=[];(OUT/'core_maps').mkdir(exist_ok=True)
with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
    activate_flyvis(device='cuda');bank,geom,bb=build_arm('C',torch,None)
    assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
    expected=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points']);assert np.array_equal(expected,geom.points)
    weights={n:v.detach().clone() for n,v in bb.net.named_parameters()}
    write('core_loaded.json',dict(device=str(bb.dev),dt=bb.dt,steps=bb.steps,fade_in_seconds=bb.fade_in_seconds,unique_reference_endpoints=len(unique),weights={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in weights.items()},scope='Same frozen native core; actual passes required, no retinal proxy.'))
    for idx,(key,ref) in enumerate(unique.items()):
        assert time.monotonic()-start<p['core_timeout_seconds'] and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
        hx=np.stack([np.load(ROOT/view['files'][0]['path'],allow_pickle=False) for view in ref['views']]).astype(np.float32);assert hx.shape==(len(ref['sequence']),721) and np.isfinite(hx).all()
        frozen=hx.copy();held=terminal_reference_history(hx);assert len(held)==len(hx)+30 and np.array_equal(held[-1],hx[-1]) and np.all(held==hx[-1])
        parity=None
        for mode,values in [('index',hx),('terminal_hold',held)]:
            status=dict(stage='core_reference',mode=mode,completed_endpoints=idx,total_endpoints=len(unique),key=key,updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat());(OUT/'core_status.json').write_bytes((json.dumps(status)+'\n').encode('utf8'))
            x=torch.from_numpy(values[None].copy()).to(bb.dev);state=bb.init(x[:,:1]);last_delta=None
            for t in range(1,len(values)):
                previous=state.nodes.activity.detach().clone() if t==len(values)-1 else None;state,_=bb.step(state,x[:,t-1],x[:,t])
                if previous is not None:last_delta=(state.nodes.activity-previous).cpu().numpy()
            value=((state.nodes.activity[0,bb.index].cpu().numpy().T.astype(float)-mu)/sd).astype(np.float32);assert value.shape==(721,34) and np.isfinite(value).all()
            if mode=='index':
                old=np.load(ROOT/ref['maps'][0]['path'],allow_pickle=False);parity=float(np.max(abs(value-old)));assert parity<=1e-4,(key,parity)
            path=OUT/'core_maps'/(key+'.'+mode+'.C34.npy')
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            records.append(dict(key=key,mode=mode,path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),input_shape=list(values.shape),original_parity=parity if mode=='index' else None,last_state_delta_max=float(abs(last_delta).max()) if last_delta is not None else None))
            assert np.array_equal(hx,frozen) and all(torch.equal(v,weights[n]) for n,v in bb.net.named_parameters())
            del x,state,value
        print('CORE_REFERENCE_COMPLETE',idx+1,len(unique),key,flush=True)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
write('core_result.json',dict(passed=True,records=records,unique_reference_endpoints=len(unique),elapsed_seconds=time.monotonic()-start,new_core_passes=2*len(unique),held_schedule='Frozen terminal_reference_history default30 repeats; same initialfade and core equations.',limitations='Finite synthetic history, not equilibrium or independent dynamics implementation. Original parity<=1e-4 is an explicit replay gate, not exact equality. Downstream original scores use archived exact maps.'))
