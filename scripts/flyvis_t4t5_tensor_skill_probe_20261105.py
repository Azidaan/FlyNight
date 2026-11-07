"""One-case decoder-input tensor intervention; recurrence and weights stay unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_t4t5_tensor_skill_probe_20261105';SRC=ROOT/'runs/flyvis_y1_latency_fp64_20261104';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'skills/neural-tensor-interventions/scripts'))
from intervention import scale_nodes
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def register():
 OUT.mkdir(exist_ok=False);prior=ROOT/'runs/flyvis_full_spatial_capture_probe_20261105';a=json.loads((prior/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'];p0=json.loads((prior/'protocol.json').read_text(encoding='utf-8'));case=p0['case'];ref=SRC/'outputs'/(case['id']+'.npz');assert sha(ref)==p0['reference_output_sha256'];crit='On a single case, identity intervention reproduces the saved decoder output within 1e-5, all outputs finite, no input/parameter mutation, four cell types per intervention, independently recomputed metrics agree. A response difference is a measurement, not a criterion of scientific success.'
 p=dict(id='flyvis_t4t5_tensor_skill_probe_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),case=case,reference_output_sha256=sha(ref),reference_protocol_sha256=sha(SRC/'protocol.json'),full_spatial_audit_sha256=sha(prior/'independent_audit.json'),skill_sha256=sha(ROOT/'skills/neural-tensor-interventions/SKILL.md'),helper_sha256=sha(ROOT/'skills/neural-tensor-interventions/scripts/intervention.py'),runner_sha256=sha(Path(__file__)),evaluator_sha256=sha(ROOT/'scripts/audit_flyvis_t4t5_tensor_skill_probe_20261105.py'),criterion=crit,arms={'reference':{'factor':1.,'types':[]},'identity_t4':{'factor':1.,'types':['T4a','T4b','T4c','T4d']},'scale_t4':{'factor':.9,'types':['T4a','T4b','T4c','T4d']},'scale_t5':{'factor':.9,'types':['T5a','T5b','T5c','T5d']},'control_other':{'factor':.9,'types':['T1','T2','T2a','T3']}},method='Apply scale only on cloned per-frame tensor activity sent to decoder, across both internal integration steps and all columns; do not mutate recurrent network state. Common unchanged initial decoded output, intervention at frames1..78. All three interventions use four cell types and 0.9 factor; this controls number, not baseline activity magnitude or biological role.',metrics='Independently compute mean raw output shift, EPE against known synthetic displacement for frames55..78, correct sign fraction, and first 3 consecutive correct sign latency with explicit missing; no optimization or parameter selection.',scope='One existing BDD TRAIN synthetic reversal sequence, stock FlyVis, CPU FP64; technical intervention harness test, not causal proof of recurrent kernel, independent dataset, tracker, or HOTA.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — техническая проверка вмешательства T4/T5 в считывание',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Меняет ли локальное вмешательство в T4/T5 вход декодера и сохраняется ли состояние сети?',hypothesis='Уменьшение сигналов на входе декодера может изменить его ответ; величина и направление эффекта заранее не известны.',expected='Проверка инструмента и независимого измерения на одном случае.',acceptance=[dict(criterion=crit,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_t4t5_tensor_skill_probe_20261105/protocol.json',claim='Протокол до пробы')]));cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered')
def run():
 import importlib,shutil,torch
 from tracking.flyvis_adapter import activate_flyvis
 from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['runner_sha256'] and sha(ROOT/'skills/neural-tensor-interventions/scripts/intervention.py')==p['helper_sha256'] and sha(ROOT/'scripts/audit_flyvis_t4t5_tensor_skill_probe_20261105.py')==p['evaluator_sha256'];refpath=SRC/'outputs'/(p['case']['id']+'.npz');assert sha(refpath)==p['reference_output_sha256'] and shutil.disk_usage(OUT).free>=5_000_000_000
 fv=activate_flyvis()
 for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
 torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
 from tracking.flyvis_motion import FlyVisMotionEstimator
 model=FlyVisMotionEstimator(mode='full',source_fps=30,flow_gain=1,temporal_resampling='hold',subtract_initial_baseline=False);model.network.double();model.decoder.double();model.eye.conv.double();net,dec=model.network,model.decoder;net.clamp();frozen=parameter_sha(net,dec);ref=np.load(refpath);hexes=ref['hexes'];inds={key:np.concatenate([np.asarray(net.stimulus.layer_index[n]) for n in v['types']]) for key,v in p['arms'].items() if v['types']};assert all(len(x)==4*721 and len(np.unique(x))==4*721 for x in inds.values());outputs={k:np.empty((79,721,2),np.float64) for k in p['arms']};unchanged=True;changed_nonselected=True;start=time.perf_counter()
 with torch.inference_mode():
  first=torch.from_numpy(hexes[0])[None,None];state=net.fade_in_state(1.,model.dt,first);init=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1)[0]*np.array([512/169,-512/169]);
  for o in outputs.values():o[0]=init
  for j in range(1,79):
   inp=torch.from_numpy(hexes[j])[None,None,None].expand(1,2,1,721);states=net.simulate(inp,model.dt,initial_state=state,as_states=True);activity=torch.stack([s.nodes.activity for s in states],dim=1);before=activity.clone()
   for name,arm in p['arms'].items():
    if name=='reference':value=activity
    else:
     indices=inds[name];value=scale_nodes(activity,indices,arm['factor']);mask=torch.ones(activity.shape[-1],dtype=torch.bool);mask[indices]=False;changed_nonselected &= bool(torch.equal(value[...,mask],activity[...,mask]));assert torch.equal(value[...,indices],activity[...,indices]*arm['factor'])
    out=dec(value)[:,-1].cpu().numpy().transpose(0,2,1)[0];outputs[name][j]=out*np.array([512/169,-512/169])
   unchanged &= bool(torch.equal(activity,before));state=states[-1]
 reference=ref['flow'][0,1];maxdiff=float(np.max(np.abs(outputs['reference']-reference)));identity=float(np.max(np.abs(outputs['identity_t4']-outputs['reference'])));passed=bool(maxdiff<=1e-5 and identity==0 and unchanged and changed_nonselected and parameter_sha(net,dec)==frozen and not torch.cuda.is_initialized() and all(np.isfinite(x).all() for x in outputs.values()));np.savez_compressed(OUT/'outputs.npz',**outputs);save(OUT/'run_result.json',dict(passed=passed,protocol_sha256=sha(OUT/'protocol.json'),outputs_sha256=sha(OUT/'outputs.npz'),reference_max_abs_diff=maxdiff,identity_max_abs_diff=identity,decoder_input_source_unchanged=unchanged,unselected_nodes_exact=changed_nonselected,parameter_sha=frozen,parameters_unchanged=parameter_sha(net,dec)==frozen,cuda_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start));print(json.dumps(dict(passed=passed,reference_max_diff=maxdiff,seconds=time.perf_counter()-start)));assert passed
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
