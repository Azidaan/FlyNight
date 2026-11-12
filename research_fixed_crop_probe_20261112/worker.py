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
from scripts.z13_target_model import build_arm,receptor_features_rich
p=verify();render=read(OUT/'render_manifest.json');assert render['passed'] and sha(OUT/'rendered_inputs.npz')==render['inputs_sha256'];rendered=np.load(OUT/'rendered_inputs.npz',allow_pickle=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=now(),stage=stage,**kw)),encoding='utf8')
norms={n:read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C','R1']}
cm,cs=[np.array(norms['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(norms['R1'][n])[:34] for n in ['mean','sd']]
maps={};inputs={};records=[];status('waiting_for_GPU_lock')
with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
    activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
    weights={n:v.detach().clone() for n,v in bb.net.named_parameters()}
    save('loaded_model.json',dict(created=now(),device=str(bb.dev),dt=bb.dt,steps=bb.steps,fade_in_seconds=bb.fade_in_seconds,weights={n:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for n,v in weights.items()},supervisor=observation.read()))
    for mode in p['modes']:
        for ci,case in enumerate(p['cases']):
            original=np.stack([np.load(ROOT/n,allow_pickle=False) for n in case['views']]).astype(np.float32)
            for ei,ep in enumerate(case['endpoints']):
                key=f'{mode}_{ci}_{ei}';hx=rendered[key].astype(np.float32);status('replaying',mode=mode,old_id=case['case']['old_id'],endpoint=ei)
                assert hx.shape==original[:ep['length']].shape and np.array_equal(hx[-1],original[ep['length']-1]) and np.isfinite(hx).all()
                frozen=hx.copy();x=torch.from_numpy(hx[None].copy()).to(bb.dev);state=bb.init(x[:,:1])
                for t in range(1,len(hx)):state,_=bb.step(state,x[:,t-1],x[:,t])
                c=((state.nodes.activity[0,bb.index].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                r=((receptor_features_rich(hx[None].astype(float))[-1][0].T-rm)/rs).astype(np.float32)
                errors={}
                for arm,array in [('C34',c),('R1',r)]:
                    assert array.shape==(721,34) and np.isfinite(array).all();maps[key+'_'+arm]=array
                    if mode=='index':
                        expected=np.load(ROOT/ep['paths'][arm],allow_pickle=False);error=float(abs(array-expected).max());errors[arm]=error;assert error<=(1e-4 if arm=='C34' else 0),(ci,ei,arm,error)
                    if mode=='identity':assert np.array_equal(array,maps[f'index_{ci}_{ei}_{arm}'])
                assert np.array_equal(hx,frozen) and all(torch.equal(v,weights[n]) for n,v in bb.net.named_parameters())
                inputs[key]=hx;records.append(dict(mode=mode,case=ci,old_id=case['case']['old_id'],endpoint=ei,length=len(hx),update_calls=len(hx)-1,changed_values=int(np.count_nonzero(hx!=original[:len(hx)])),parity_errors=errors,weights_unchanged=True,terminal_input_unchanged=True))
                del x,state;print('REPLAY',mode,case['case']['old_id'],ei,flush=True)
        if mode=='index':save('baseline_gate.json',dict(passed=True,records=records.copy(),all_original_cases_checked_before_intervention=True))
verify()
with (OUT/'maps.npz').open('xb') as f:np.savez_compressed(f,**maps)
with (OUT/'expanded_inputs.npz').open('xb') as f:np.savez_compressed(f,**inputs)
save('replay.json',dict(passed=True,created=now(),records=records,maps_sha256=sha(OUT/'maps.npz'),expanded_inputs_sha256=sha(OUT/'expanded_inputs.npz'),weights_unchanged=True,tracker_calls=0))
status('replay_complete');print('REPLAY_COMPLETE',flush=True)
