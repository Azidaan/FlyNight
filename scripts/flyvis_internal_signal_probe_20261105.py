"""Prospective, low-capacity linear probe on already saved mean activities."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_internal_signal_probe_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def register():
 OUT.mkdir(exist_ok=False);prior=json.loads((SRC/'independent_audit.json').read_text(encoding='utf-8'));assert prior['passed'] and prior['report_sha256']==sha(SRC/'report.json');cases=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'))['cases'];crit='On each stock/fast x normal/low contrast x still/moving start evaluation group, T4T5 fitted linear readout has reversal-frame EPE at most 90% of matched fitted decoder readout AND strictly less than no-FlyVis phase correlation; report direction and latency as secondary, not causal proof.'
 p=dict(id='flyvis_internal_signal_probe_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),source_protocol_sha256=sha(SRC/'protocol.json'),source_report_sha256=sha(SRC/'report.json'),source_audit_sha256=sha(SRC/'independent_audit.json'),source_inference_sha256=sha(ROOT/'scripts/flyvis_y1_latency_fp64_20261104.py'),self_sha256=sha(Path(__file__)),cases=len(cases),criterion=crit,stages=['R1','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'],definition='Saved activity[0] after last state at each frame, average of central mask for each cell type; subtract same stage at frame 0. Saved flow[0,1] is decoder hold-last, central mean, subtract frame 0. No extra network inference. All readout inputs are current and past frames only; target velocity only for fitting and offline evaluation.',heads={'R1':[0],'T4':[1,2,3,4],'T5':[5,6,7,8],'T4T5':[1,2,3,4,5,6,7,8],'all9':list(range(9)),'decoder':'two decoder outputs'},fitting='For each stock/fast and each head fit 2D linear readout with no intercept on calibration scenes, frames31..78 for both starts, contrasts, axes and speeds; ridge lambda=1e-4 * trace(X.T@X)/dimension, fixed prior to analysis, float64; zero-design head returns zero. No threshold or feature selection on evaluation. Head sizes differ; T4T5 vs decoder is not equal-capacity/causal ablation.',evaluation='Per 4 evaluation scenes x 2 speeds x 2 axes =16 cases for each start/contrast; mean vector EPE on 24 frames after reverse55..78 and before31..54; correct sign fraction and first three consecutive correct sign reversal latency, number identified; include fitted decoder, zero and phase correlation on matching frames. Bootstrap/independence not asserted.',scope='All scenes and saved outputs previously studied, BDD training videos, one fly model and 2 time-constant settings; synthetic global motion, not independent test and no full object tracking/HOTA. Activity averaged spatially; cannot infer individual-neuron mechanism or prove no encoded information. Phase control best suited global shifts.',sources=['https://www.nature.com/articles/s41586-024-07939-3'],baseline='Comparing differing-size linear probes tests access to information under this representation, not contribution of neuronal architecture. No inference, model changes or GT during prediction.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — где доступен сигнал при развороте',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Содержат ли усреднённые сигналы R1 и T4/T5 полезную для разворота информацию относительно выхода декодера?',hypothesis='Доступность информации при ограниченном линейном считывании может различаться между усреднёнными группами и готовым выходом; пространственное усреднение и малое число сцен ограничивают интерпретацию.',expected='Исследовательская диагностика по сохранённым активностям без запуска сети.',acceptance=[dict(criterion=crit,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_internal_signal_probe_20261105/protocol.json',claim='Протокол до расчёта')]));cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['self_sha256'] and sha(SRC/'protocol.json')==p['source_protocol_sha256'];original=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));mask=np.load(SRC/'geometry.npz')['mask'];cv2.setNumThreads(2);data=[];start=time.perf_counter()
 for c in original['cases']:
  inp=SRC/'inputs'/(c['id']+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256'];frames=np.load(inp)['frames'];x={}
  for arm in ('stock','fast'):
   path=SRC/arm/'outputs'/(c['id']+'.npz');m=json.loads(path.with_suffix('.json').read_text());assert sha(path)==m['sha256'] and m['input_sha256']==im['sha256'] and m['protocol_sha256']==p['source_protocol_sha256'];z=np.load(path);a=z['activity'][0];q=z['flow'][0,1][:,mask].mean(1);assert a.shape==(79,9) and np.isfinite(a).all() and np.isfinite(q).all();x[arm]=dict(activity=a-a[0],decoder=q-q[0])
  phase=[];invalid=0
  if c['split']=='evaluation':
   for t in range(55,79):
    a=frames[t-1].astype(np.float32);b=frames[t].astype(np.float32)
    if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));invalid+=1;continue
    v,_=cv2.phaseCorrelate(a,b)
    if not np.isfinite(v).all():phase.append((0.,0.));invalid+=1
    else:phase.append(v)
  data.append(dict(case=c,x=x,truth=positions(c)[1].astype(float),phase=np.asarray(phase) if phase else None,phase_invalid=invalid))
  if len(data)%16==0:save(OUT/'status.json',dict(stage='load',done=len(data),total=len(original['cases']),elapsed=time.perf_counter()-start))
 def vector(item,arm,head):
  z=item['x'][arm];return z['decoder'] if head=='decoder' else z['activity'][:,p['heads'][head]]
 weights={}
 for arm in ('stock','fast'):
  weights[arm]={}
  for head in p['heads']:
   X=np.concatenate([vector(i,arm,head)[31:79] for i in data if i['case']['split']=='calibration']);Y=np.concatenate([i['truth'][31:79] for i in data if i['case']['split']=='calibration']);gram=X.T@X;ridge=float(np.trace(gram)/gram.shape[0]*1e-4);W=np.linalg.solve(gram+np.eye(gram.shape[0])*ridge,X.T@Y) if ridge>0 else np.zeros((gram.shape[0],2));assert np.isfinite(W).all();weights[arm][head]=dict(coefficients=W.tolist(),ridge=ridge,train_X_rank=int(np.linalg.matrix_rank(X)))
 rows=[]
 for item in data:
  c=item['case'];y=item['truth'];res={}
  for arm in ('stock','fast'):
   for head in p['heads']:
    q=vector(item,arm,head)@np.asarray(weights[arm][head]['coefficients']);pred=q[55:79,c['axis']]<0;hit=next((j for j in range(22) if pred[j:j+3].all()),None);res[arm+'_'+head]=dict(after_epe=float(np.linalg.norm(q[55:79]-y[55:79],axis=1).mean()),before_epe=float(np.linalg.norm(q[31:55]-y[31:55],axis=1).mean()),after_correct=float(np.mean(pred)),latency=hit)
  res['zero']=dict(after_epe=float(np.linalg.norm(y[55:79],axis=1).mean()))
  if item['phase'] is not None:res['phase']=dict(after_epe=float(np.linalg.norm(item['phase']-y[55:79],axis=1).mean()),invalid=item['phase_invalid'])
  rows.append(dict(case=c,results=res))
 groups={}
 for split in ('calibration','evaluation'):
  for startkind in ('still','moving'):
   for contrast in (1.,.25):
    cohort=[x for x in rows if x['case']['split']==split and x['case']['start']==startkind and x['case']['contrast']==contrast];assert len(cohort)==16;summary={}
    for name in cohort[0]['results']:
     arr=[x['results'][name] for x in cohort];q={'after_epe':float(np.mean([x['after_epe'] for x in arr]))}
     if name not in ('zero','phase'):
      lat=[x['latency'] for x in arr if x['latency'] is not None];q.update(before_epe=float(np.mean([x['before_epe'] for x in arr])),after_correct=float(np.mean([x['after_correct'] for x in arr])),latency_identified=len(lat),latency_mean=float(np.mean(lat)) if len(lat)==len(arr) else None)
     if name=='phase':q['invalid']=sum(x['invalid'] for x in arr)
     summary[name]=q
    groups[f'{split}_{startkind}_{contrast}']=summary
 checks={}
 for arm in ('stock','fast'):
  for st in ('still','moving'):
   for ct in (1.,.25):
    g=groups[f'evaluation_{st}_{ct}'];v=g[arm+'_T4T5']['after_epe'];checks[f'{arm}_{st}_{ct}_vs_decoder']=v<=.9*g[arm+'_decoder']['after_epe'];checks[f'{arm}_{st}_{ct}_vs_phase']=v<g['phase']['after_epe']
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),weights=weights,groups=groups,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start,scope=p['scope']));print(json.dumps(dict(groups={k:v for k,v in groups.items() if k.startswith('evaluation')},checks=checks,seconds=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
