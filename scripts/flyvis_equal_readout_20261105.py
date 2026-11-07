"""Compare fixed two-dimensional T4/T5 projection and 2D decoder on separate saved scenes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_equal_readout_20261105';TRAIN=ROOT/'runs/flyvis_offset_recovery2_20261104';STOCK=ROOT/'runs/flyvis_y1_latency_fp64_20261104';FAST=ROOT/'runs/flyvis_tau_half_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions as train_positions
from scripts.flyvis_y1_latency_fp64_20261104 import positions as eval_positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def register():
 OUT.mkdir(exist_ok=False);prior=ROOT/'runs/flyvis_internal_signal_probe_20261105';a=json.loads((prior/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'] and a['report_sha256']==sha(prior/'report.json');tr=json.loads((TRAIN/'protocol.json').read_text(encoding='utf-8'));ev=json.loads((STOCK/'protocol.json').read_text(encoding='utf-8'));tf=json.loads((FAST/'protocol.json').read_text(encoding='utf-8'));assert not set(tr['videos'])&set(ev['videos']) and ev['videos']==tf['videos'];assert json.loads((STOCK/'independent_audit.json').read_text(encoding='utf-8'))['passed'];assert json.loads((FAST/'chain_result.json').read_text(encoding='utf-8'))['complete'];assert json.loads((FAST/'final_verification.json').read_text(encoding='utf-8'))['streaming_reference_passed']
 crit='For each stock/fast and both contrasts, evaluation reverse frames55..78: equal-input T4T5 projection EPE <=90% of calibrated decoder EPE, reversal latency mean no worse than decoder and latency identified for all 16. Invalid latency fails. Phase correlation reported as no-FlyVis comparator, not primary acceptance.'
 p=dict(id='flyvis_equal_readout_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),source_train_protocol_sha256=sha(TRAIN/'protocol.json'),source_train_audit_sha256=sha(TRAIN/'independent_audit.json'),source_stock_protocol_sha256=sha(STOCK/'protocol.json'),source_stock_audit_sha256=sha(STOCK/'independent_audit.json'),source_fast_protocol_sha256=sha(FAST/'protocol.json'),source_fast_chain_sha256=sha(FAST/'chain_result.json'),source_prior_report_sha256=sha(prior/'report.json'),source_prior_audit_sha256=sha(prior/'independent_audit.json'),self_sha256=sha(Path(__file__)),criterion=crit,method='For every time point subtract frame-zero signal; project 8 mean T4/T5 activities onto fixed data-independent 2D contrasts [T4a+T5a-T4c-T5c, T4b+T5b-T4d-T5d] /2. Do not claim direction preference of indices. Compare against 2D saved decoder output. Both downstream readouts have 2 inputs, 2 outputs, 4 supervised coefficients, no intercept, same ridge = 1e-4 trace(XTX)/2; separately trained for stock and fast. Secondary 8-input T4T5 (16 weights) included to reveal loss of fixed projection. Fixed projection is an architectural choice and can remove relevant information; equal supervised weights do not make the representations equivalent.',split='Training: all 8 previously used flyvis_offset_recovery2 scenes and both starts. Evaluation: four different BDD TRAIN images used in the older y1 latency/tau half experiments, only reverse events; no overlap with 8. These 4 were studied earlier and all 12 scenes belong to original training videos, so not a new independent final test. No fitting using evaluation.',metrics='Per stock/fast x contrast evaluation group: 16 cases (4 scenes x 2 axes x 2 speeds), mean EPE before31..54 and after55..78, correct direction after reverse, time to first three consecutive correct signs and number identified. No-FlyVis phase correlation and zero control evaluated on matching reverse frames.',scope='Same frozen precomputed FP64 signals: 1 model and two tau settings; spatial average of 9 types discards topology; synthetic global motion, not natural video/tracker/HOTA. Ground truth only fit and evaluation; no run of FlyVis, no GT during prediction, all inputs causal at frame t.',source=['https://www.nature.com/articles/s41586-024-07939-3'],sources_scope='The paper motivates investigating T4/T5, not the specific fixed projection or claims of decoder failure.')
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(c['id']==p['id'] for c in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — считыватели с двумя входами на других сценах',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Сохраняется ли частное преимущество T4/T5 при одинаковом числе входов и на других сценах?',hypothesis='Двум фиксированным комбинациям T4/T5 может быть доступна информация для разворота, но выбор проекции может её потерять; сигнал декодера может содержать другую информацию.',expected='Проверка по уже сохранённым активностям, без нового прогона FlyVis.',acceptance=[dict(criterion=crit,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_equal_readout_20261105/protocol.json',claim='Протокол до расчёта')]))
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['self_sha256'];assert sha(TRAIN/'protocol.json')==p['source_train_protocol_sha256'] and sha(STOCK/'protocol.json')==p['source_stock_protocol_sha256'] and sha(FAST/'protocol.json')==p['source_fast_protocol_sha256'];cv2.setNumThreads(2);start=time.perf_counter();train_cases=json.loads((TRAIN/'protocol.json').read_text(encoding='utf-8'))['cases'];eval_cases=[c for c in json.loads((STOCK/'protocol.json').read_text(encoding='utf-8'))['cases'] if c['event']=='reverse'];masktr=np.load(TRAIN/'geometry.npz')['mask'];maskev=np.load(STOCK/'geometry.npz')['mask'];assert np.array_equal(maskev,np.load(FAST/'geometry.npz')['mask']);items=[]
 def load_case(c,is_train):
  base=TRAIN if is_train else STOCK;inp=base/'inputs'/(c['id']+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256'];f=np.load(inp)['frames'];v={};mask=masktr if is_train else maskev
  for arm in ('stock','fast'):
   directory=(TRAIN/arm if is_train else (STOCK if arm=='stock' else FAST));path=directory/'outputs'/(c['id']+'.npz');meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['sha256'] and meta['input_sha256']==im['sha256'] and meta['protocol_sha256']==sha(directory/'protocol.json');z=np.load(path);act=z['activity'][0];flow=z['flow'][0,1][:,mask].mean(1);assert np.isfinite(act).all() and np.isfinite(flow).all();a=act-act[0];v[arm]=dict(projected=np.column_stack(((a[:,1]+a[:,5]-a[:,3]-a[:,7])*.5,(a[:,2]+a[:,6]-a[:,4]-a[:,8])*.5)),decoder=flow-flow[0],all8=a[:,1:9])
  return f,v
 for c in train_cases:
  frames,v=load_case(c,True);items.append(dict(case=c,split='train',features=v,truth=train_positions(c)[1].astype(float),phase=None))
 for c in eval_cases:
  frames,v=load_case(c,False);truth=eval_positions(c)[1].astype(float);phase=[];invalid=0
  for t in range(55,79):
   x=frames[t-1].astype(np.float32);y=frames[t].astype(np.float32)
   if np.ptp(x)==0 or np.ptp(y)==0:phase.append((0.,0.));invalid+=1;continue
   xy,_=cv2.phaseCorrelate(x,y)
   if not np.isfinite(xy).all():phase.append((0.,0.));invalid+=1
   else:phase.append(xy)
  items.append(dict(case=c,split='evaluation',features=v,truth=truth,phase=np.asarray(phase),invalid=invalid))
 weights={}
 for arm in ('stock','fast'):
  weights[arm]={}
  for head in ('projected','decoder','all8'):
   X=np.concatenate([z['features'][arm][head][31:79] for z in items if z['split']=='train']);Y=np.concatenate([z['truth'][31:79] for z in items if z['split']=='train']);gram=X.T@X;ridge=float(np.trace(gram)/gram.shape[0]*1e-4);W=np.linalg.solve(gram+np.eye(gram.shape[0])*ridge,X.T@Y) if ridge>0 else np.zeros((gram.shape[0],2));assert np.isfinite(W).all();weights[arm][head]=dict(coefficients=W.tolist(),ridge=ridge,rank=int(np.linalg.matrix_rank(X)))
 rows=[]
 for z in items:
  if z['split']!='evaluation':continue
  c=z['case'];truth=z['truth'];res={}
  for arm in ('stock','fast'):
   for head in ('projected','decoder','all8'):
    y=z['features'][arm][head]@np.asarray(weights[arm][head]['coefficients']);sign=y[55:79,c['axis']]<0;hit=next((t for t in range(22) if sign[t:t+3].all()),None);res[arm+'_'+head]=dict(after_epe=float(np.linalg.norm(y[55:79]-truth[55:79],axis=1).mean()),before_epe=float(np.linalg.norm(y[31:55]-truth[31:55],axis=1).mean()),after_correct=float(np.mean(sign)),latency=hit)
  res['phase']=dict(after_epe=float(np.linalg.norm(z['phase']-truth[55:79],axis=1).mean()),invalid=z['invalid']);res['zero']=dict(after_epe=float(np.linalg.norm(truth[55:79],axis=1).mean()));rows.append(dict(case=c,results=res))
 groups={}
 for contrast in (1.,.25):
  cohort=[z for z in rows if z['case']['contrast']==contrast];assert len(cohort)==16;summary={}
  for head in cohort[0]['results']:
   a=[z['results'][head] for z in cohort];v=dict(after_epe=float(np.mean([x['after_epe'] for x in a])))
   if head not in ('zero','phase'):
    lat=[x['latency'] for x in a if x['latency'] is not None];v.update(before_epe=float(np.mean([x['before_epe'] for x in a])),after_correct=float(np.mean([x['after_correct'] for x in a])),latency_identified=len(lat),latency_mean=float(np.mean(lat)) if len(lat)==len(a) else None)
   if head=='phase':v['invalid']=sum(x['invalid'] for x in a)
   summary[head]=v
  groups[str(contrast)]=summary
 checks={}
 for arm in ('stock','fast'):
  for ct in (1.,.25):
   z=groups[str(ct)];a=z[arm+'_projected'];b=z[arm+'_decoder'];checks[f'{arm}_{ct}_epe']=a['after_epe']<=.9*b['after_epe'];checks[f'{arm}_{ct}_latency']=a['latency_mean'] is not None and b['latency_mean'] is not None and a['latency_mean']<=b['latency_mean']
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),weights=weights,groups=groups,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start,scope=p['scope']));print(json.dumps(dict(groups=groups,checks=checks,seconds=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
