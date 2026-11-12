import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-12T07:40:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
import numpy as np
import torch
from tracking.gpu_lock import GpuLock
from tracking.flyvis_adapter import activate_flyvis
from tracking.flyvis_activity import OUTPUT_CELL_TYPES
from scripts.z13_target_model import build_arm,receptor_features_rich
p=verify();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=now(),stage=stage,**kw)),encoding='utf8')
norms={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']}
cm,cs=[np.array(norms['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(norms['R1'][n])[:34] for n in ['mean','sd']]
maps={};inputs={};states={};records=[];status('waiting_for_GPU_lock')
with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
    activate_flyvis(device='cuda');bank,geom,bb=build_arm('C',torch,None)
    assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
    weights={n:v.detach().clone() for n,v in bb.net.named_parameters()}
    # Audit actual loaded indexes, not just a matching saved connectome directory.
    conn=bb.net.connectome;u=np.asarray(conn.nodes.u[:]);v=np.asarray(conn.nodes.v[:]);types=np.asarray(conn.nodes.type[:]).astype(str)
    uv=np.array([(a,b) for a in range(-15,16) for b in range(max(-15,-15-a),min(15,15-a)+1)])
    expected_xy=np.c_[13*uv[:,1],np.trunc(13*(uv[:,0]+uv[:,1]/2))]+195
    idx=bb.index.cpu().numpy();input_idx=np.asarray(bb.net.stimulus.input_index);checks=[]
    for family,labels,indexes in [('output',OUTPUT_CELL_TYPES,idx),('input',conn.input_cell_types[:].astype(str),input_idx)]:
        for name,ix in zip(labels,indexes):
            coords=np.c_[u[ix],v[ix]];assert np.array_equal(coords,uv) and np.all(types[ix]==name)
            assert len(np.unique(ix))==721;checks.append(dict(family=family,type=name,cells=len(ix),exact=True))
    saved=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'])
    assert np.array_equal(expected_xy,geom.points) and np.array_equal(saved,geom.points)
    save('coordinate_audit.json',dict(passed=True,created=now(),connectome_path=str(conn.path),checks=checks,saved_geometry_exact=True,nodes=len(u),index_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),scope='Coordinate assignment only; does not prove physical correspondence between two changing object crops.'))
    save('loaded_model.json',dict(created=now(),device=str(bb.dev),dt=bb.dt,steps=bb.steps,fade_in_seconds=bb.fade_in_seconds,weights={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in weights.items()},supervisor=observation.read()))
    for mode in p['modes']:
        for ci,case in enumerate(p['cases']):
            original=np.stack([np.load(ROOT/n,allow_pickle=False) for n in case['views']]).astype(np.float32)
            for ei,ep in enumerate(case['endpoints']):
                key=f'{mode}_{ci}_{ei}';hx=original[:ep['length']].copy()
                if mode=='terminal_hold':hx[:]=hx[-1].copy()
                if mode.endswith('_hold'):hx=np.concatenate([hx,np.repeat(hx[-1:],30,axis=0)])
                status('replaying',mode=mode,old_id=case['case']['old_id'],endpoint=ei)
                assert np.array_equal(hx[-1],original[ep['length']-1]) and np.isfinite(hx).all()
                frozen=hx.copy();x=torch.from_numpy(hx[None].copy()).to(bb.dev);state=bb.init(x[:,:1]);last_residual=None
                for t in range(1,len(hx)):
                    previous=state.nodes.activity.detach().clone() if t==len(hx)-1 else None
                    state,_=bb.step(state,x[:,t-1],x[:,t])
                    if previous is not None:last_residual=(state.nodes.activity-previous).cpu().numpy()
                full=state.nodes.activity.cpu().numpy().copy();states[key]=full;states[key+'_last_delta']=last_residual
                c=((state.nodes.activity[0,bb.index].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                r=((receptor_features_rich(hx[None].astype(float))[-1][0].T-rm)/rs).astype(np.float32)
                errors={}
                for arm,array in [('C34',c),('R1',r)]:
                    assert array.shape==(721,34) and np.isfinite(array).all();maps[key+'_'+arm]=array
                    if mode=='index':
                        expected=np.load(ROOT/ep['paths'][arm],allow_pickle=False);error=float(abs(array-expected).max());errors[arm]=error;assert error<=(1e-4 if arm=='C34' else 0),(ci,ei,arm,error)
                    if mode=='identity':assert np.array_equal(array,maps[f'index_{ci}_{ei}_{arm}'])
                assert np.array_equal(hx,frozen) and all(torch.equal(v,weights[n]) for n,v in bb.net.named_parameters())
                inputs[key]=hx;records.append(dict(mode=mode,case=ci,old_id=case['case']['old_id'],endpoint=ei,length=len(hx),update_calls=len(hx)-1,parity_errors=errors,weights_unchanged=True,terminal_input_unchanged=True,last_state_delta_max=float(abs(last_residual).max()),last_state_delta_rms=float(np.sqrt(np.mean(last_residual.astype(float)**2)))))
                del x,state;print('REPLAY',mode,case['case']['old_id'],ei,flush=True)
        if mode=='index':save('baseline_gate.json',dict(passed=True,records=records.copy(),all_original_cases_checked_before_intervention=True))
verify()
for name,value in [('maps',maps),('expanded_inputs',inputs),('states',states)]:
    with (OUT/(name+'.npz')).open('xb') as f:np.savez_compressed(f,**value)
contrasts=[]
for ci,case in enumerate(p['cases']):
    for ei in [0,1]:
        delta=states[f'history_hold_{ci}_{ei}']-states[f'terminal_hold_{ci}_{ei}'];contrasts.append(dict(old_id=case['case']['old_id'],endpoint=ei,max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta.astype(float)**2)))))
save('replay.json',dict(passed=True,created=now(),records=records,history_contrasts=contrasts,maps_sha256=sha(OUT/'maps.npz'),expanded_inputs_sha256=sha(OUT/'expanded_inputs.npz'),states_sha256=sha(OUT/'states.npz'),weights_unchanged=True,tracker_calls=0))
status('replay_complete');print('REPLAY_COMPLETE',flush=True)
