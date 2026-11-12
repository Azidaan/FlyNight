import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
deadline=datetime.fromisoformat('2026-11-12T18:15:00+03:00');remaining=(deadline-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=min(1800,remaining))
import numpy as np
import torch
from tracking.gpu_lock import GpuLock
from tracking.flyvis_adapter import activate_flyvis
from scripts.z13_target_model import build_arm,receptor_features_rich
p=verify();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def status(stage,**kwargs):(OUT/'status.json').write_text(json.dumps(dict(updated=now(),stage=stage,**kwargs)),encoding='utf8')
render_manifest=read(OUT/'render_manifest.json');assert render_manifest['passed'] and sha(OUT/'expanded_inputs.npz')==render_manifest['inputs_sha256']
rendered=np.load(OUT/'expanded_inputs.npz',allow_pickle=False)
def expanded(case,mode):
    ci=next(i for i,c in enumerate(p['cases']) if c['case']['old_id']==case['case']['old_id'])
    hx=rendered[f'{mode}_{ci}'].copy();return hx,list(range(len(hx)))
norms={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']}
cm,cs=[np.array(norms['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(norms['R1'][n])[:34] for n in ['mean','sd']]
results=[];maps={};schedule=[];status('waiting_for_GPU_lock')
with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
    activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
    weights={n:v.detach().clone() for n,v in bb.net.named_parameters()}
    save('loaded_model.json',dict(created=now(),device=str(bb.dev),dt=bb.dt,steps=bb.steps,fade_in_seconds=bb.fade_in_seconds,weights={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in weights.items()},index_sha256=hashlib.sha256(bb.index.cpu().numpy().tobytes()).hexdigest(),supervisor=observation.read()))
    # Gate all untouched cases before running any changed history.
    for mode in p['modes']:
        for ci,case in enumerate(p['cases']):
            status('replaying',mode=mode,case=case['case']['old_id']);hx,positions=expanded(case,mode);frozen=hx.copy();x=torch.from_numpy(hx[None].copy()).to(bb.dev)
            wanted={positions[ep['length']-1]:ei for ei,ep in enumerate(case['endpoints'])};assert len(wanted)==2
            state=bb.init(x[:,:1]);ordinary=receptor_features_rich(hx[None].astype(float));captured={}
            for t in range(len(hx)):
                if t:state,_=bb.step(state,x[:,t-1],x[:,t])
                if t in wanted:
                    ei=wanted[t];activity=state.nodes.activity[:,bb.index]
                    captured[ei]=dict(C34=((activity[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32),R1=((ordinary[t][0].T-rm)/rs).astype(np.float32))
            assert np.array_equal(hx,frozen)
            key=f'{mode}_{ci}';schedule.append(dict(mode=mode,case=ci,old_id=case['case']['old_id'],positions=positions,frame_span=case['sequence'][-1][0]-case['sequence'][0][0],update_calls=len(hx)-1))
            errors={}
            for ei,value in captured.items():
                for arm,array in value.items():
                    assert array.shape==(721,34) and np.isfinite(array).all();maps[f'{key}_{ei}_{arm}']=array
                    if mode=='index':
                        expected=np.load(ROOT/case['endpoints'][ei]['paths'][arm],allow_pickle=False);error=float(abs(array-expected).max());errors[f'{ei}_{arm}']=error
                        assert error<=(1e-4 if arm=='C34' else 0),(ci,ei,arm,error)
                    if mode=='identity':assert np.array_equal(array,maps[f'index_{ci}_{ei}_{arm}'])
            assert all(torch.equal(v,weights[n]) for n,v in bb.net.named_parameters())
            results.append(dict(mode=mode,case=ci,old_id=case['case']['old_id'],parity_errors=errors,inputs_unchanged=True,weights_unchanged=True))
            del x,state,ordinary,captured
            print('REPLAY',mode,case['case']['old_id'],'updates',len(hx)-1,flush=True)
        if mode=='index':save('baseline_gate.json',dict(passed=True,records=results.copy(),all_original_cases_checked_before_intervention=True))
verify()
with (OUT/'maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
save('replay.json',dict(passed=True,created=now(),records=results,schedules=schedule,maps_sha256=sha(OUT/'maps.npz'),expanded_inputs_sha256=sha(OUT/'expanded_inputs.npz'),weights_unchanged=True,tracker_calls=0))
status('replay_complete');print('REPLAY_COMPLETE',flush=True)
