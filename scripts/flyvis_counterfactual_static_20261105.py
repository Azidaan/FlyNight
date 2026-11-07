"""Replay a held-first-frame counterfactual for frames 1..30 using saved network outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_counterfactual_static_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def register():
 OUT.mkdir(exist_ok=False);old=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));a=json.loads((SRC/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'] and a['report_sha256']==sha(SRC/'report.json');prior=json.loads((ROOT/'runs/flyvis_still_gate_20261105/independent_audit.json').read_text(encoding='utf-8'));assert prior['passed']
 crit='For stock and fast, for each contrast, on evaluation moving-start EPE over frames 1:30 counterfactual subtraction must be strictly less than both raw and first-frame subtraction, and <= phase correlation; still-start EPE must not exceed raw/first. No latency or generalisation claim. This is an exploratory stopping criterion on previously observed scenes.'
 p=dict(id='flyvis_counterfactual_static_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),source_protocol_sha256=sha(SRC/'protocol.json'),source_report_sha256=sha(SRC/'report.json'),source_audit_sha256=sha(SRC/'independent_audit.json'),source_script_sha256=sha(SRC/'../flyvis_still_gate_20261105.py') if (SRC/'../flyvis_still_gate_20261105.py').exists() else sha(ROOT/'scripts/flyvis_still_gate_20261105.py'),self_sha256=sha(Path(__file__)),criterion=crit,concept='For each actual sequence pair with identical initial frame, subtract at each timestep the existing network output from its matching still-start sequence. Its first 31 frames must be identical to its first frame, proving it represents a causal parallel FlyVis state driven by the held first frame through t=30. This is our proposed counterfactual output comparison, not a method claimed by the cited article. Reuses saved CPU FP64 outputs, no network inference or ground truth at prediction.',comparison='Per stock/fast compare raw, subtract first decoded baseline, subtract dynamic held-frame output. Separate scalar least-squares gain per version/method fit only on four calibration images, moving+still frames 1..30; no tuning on evaluation. Zero and uncalibrated phase correlation are no-FlyVis controls; phase on each adjacent image, quality failure to zero and count. Previous implementations reset time at frame21 and used frames21..78; do not compare unlike windows.',scope='Existing eight BDD training-video first frames and synthetic global translation, previously inspected/evaluated; no newly independent dataset, no real objects, no HOTA, no guarantee beyond first 30 frames. Static branch for still-start shares its own saved output and hence its zero residual is tautological. Actual parallel compute cost not measured. Contrast and appearance change after first frame are not addressed.',scientific_basis='https://doi.org/10.1073/pnas.2002937117 : primary evidence visual motion circuits can respond to static edges; not evidence this proposed subtraction works or is biologically implemented.',source_cases=len(old['cases']))
 save(OUT/'protocol.json',p);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==p['id'] for x in d['experiments']);d['experiments'].append(dict(id=p['id'],title='FlyVis — ответ на удерживаемый первый кадр как контроль смещения',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Улучшает ли вычитание динамического ответа на удерживаемый первый кадр оценку движения с самого начала?',hypothesis='Параллельное состояние сети на постоянном первом кадре может оценить временной дрейф отклика, не используя будущие кадры и не предполагая реального покоя; изменение сцены и нелинейность могут нарушить предположение.',expected='Исследовательский повтор по сохранённым выходам без запуска сети.',acceptance=[dict(criterion=crit,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_counterfactual_static_20261105/protocol.json',claim='Протокол до анализа')]));cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered')
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['self_sha256'] and sha(SRC/'protocol.json')==p['source_protocol_sha256'] and sha(SRC/'report.json')==p['source_report_sha256'];old=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));cases=old['cases'];mask=np.load(SRC/'geometry.npz')['mask'];items=[];cv2.setNumThreads(2);start=time.perf_counter();static_cache={}
 def load_case(case):
  name=case['id'];inp=SRC/'inputs'/(name+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256'];f=np.load(inp)['frames'];outs={}
  for arm in ('stock','fast'):
   path=SRC/arm/'outputs'/(name+'.npz');m=json.loads(path.with_suffix('.json').read_text());assert sha(path)==m['sha256'] and m['input_sha256']==im['sha256'] and m['protocol_sha256']==p['source_protocol_sha256'];q=np.load(path)['flow'][0,1][:,mask].mean(1);assert np.isfinite(q).all();outs[arm]=q
  return f,outs
 for c in cases:
  key=(c['scene'],c['contrast'],c['speed'],c['axis']);
  if c['start']=='still':static_cache[key]=load_case(c)
 assert len(static_cache)*2==len(cases)
 for c in cases:
  key=(c['scene'],c['contrast'],c['speed'],c['axis']);ref_frame,ref=static_cache[key];assert np.array_equal(ref_frame[:31],np.repeat(ref_frame[0:1],31,axis=0));frames,q=(ref_frame,ref) if c['start']=='still' else load_case(c);assert np.array_equal(frames[0],ref_frame[0]);vals={}
  for arm in ('stock','fast'):
   vals[arm+'_raw']=q[arm][1:31];vals[arm+'_first']=q[arm][1:31]-q[arm][0];vals[arm+'_counter']=q[arm][1:31]-ref[arm][1:31]
  truth=positions(c)[1][1:31].astype(float);phase=None;bad=0
  if c['split']=='evaluation':
   phase=[]
   for t in range(1,31):
    x=frames[t-1].astype(np.float32);y=frames[t].astype(np.float32)
    if np.ptp(x)==0 or np.ptp(y)==0:phase.append((0.,0.));bad+=1;continue
    v,quality=cv2.phaseCorrelate(x,y)
    if not np.isfinite(v).all():phase.append((0.,0.));bad+=1
    else:phase.append(v)
   phase=np.asarray(phase)
  items.append(dict(case=c,values=vals,truth=truth,phase=phase,phase_invalid=bad))
 gains={}
 for name in items[0]['values']:
  x=np.concatenate([i['values'][name].ravel() for i in items if i['case']['split']=='calibration']);y=np.concatenate([i['truth'].ravel() for i in items if i['case']['split']=='calibration']);assert x@x>0;g=float(x@y/(x@x));assert g>0;gains[name]=g
 rows=[]
 for i in items:
  metrics={}
  for name,v in i['values'].items():
   q=v*gains[name];metrics[name]=dict(epe=float(np.linalg.norm(q-i['truth'],axis=1).mean()),positive_fraction=float(np.mean(q[:,i['case']['axis']]>0)))
  metrics['zero']=dict(epe=float(np.linalg.norm(i['truth'],axis=1).mean()))
  if i['phase'] is not None:metrics['phase']=dict(epe=float(np.linalg.norm(i['phase']-i['truth'],axis=1).mean()),invalid=i['phase_invalid'])
  rows.append(dict(case=i['case'],metrics=metrics))
 groups={}
 for split in ('calibration','evaluation'):
  for startkind in ('still','moving'):
   for contrast in (1.,.25):
    cohort=[x for x in rows if x['case']['split']==split and x['case']['start']==startkind and x['case']['contrast']==contrast];assert len(cohort)==16;groups[f'{split}_{startkind}_{contrast}']={name:dict(epe=float(np.mean([z['metrics'][name]['epe'] for z in cohort]))) for name in cohort[0]['metrics']}
 checks={}
 for arm in ('stock','fast'):
  for contrast in (1.,.25):
   m=groups[f'evaluation_moving_{contrast}'];s=groups[f'evaluation_still_{contrast}'];k=arm+'_counter';checks[f'{arm}_{contrast}_moving_lt_raw']=m[k]['epe']<m[arm+'_raw']['epe'];checks[f'{arm}_{contrast}_moving_lt_first']=m[k]['epe']<m[arm+'_first']['epe'];checks[f'{arm}_{contrast}_moving_le_phase']=m[k]['epe']<=m['phase']['epe'];checks[f'{arm}_{contrast}_still_le_raw_first']=s[k]['epe']<=min(s[arm+'_raw']['epe'],s[arm+'_first']['epe'])
 save(OUT/'report.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),gains=gains,groups=groups,checks=checks,preliminary_gate=all(checks.values()),rows=rows,elapsed_seconds=time.perf_counter()-start,scope=p['scope']));print(json.dumps(dict(gains=gains,groups={k:v for k,v in groups.items() if k.startswith('evaluation')},checks=checks,seconds=time.perf_counter()-start),ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
