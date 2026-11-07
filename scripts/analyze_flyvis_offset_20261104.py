"""Раздельная подгонка на 4 сценах и оценка на 4 новых сценах."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_20261104 import positions
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));mask=np.load(OUT/'geometry.npz')['mask'];data=[]
 for arm in ('stock','fast'):assert json.loads((OUT/arm/'inference_completed.json').read_text())['complete']
 for case in p['cases']:
  values={};input_path=OUT/'inputs'/(case['id']+'.npz');meta=json.loads(input_path.with_suffix('.json').read_text())
  assert sha(input_path)==meta['sha256']
  for arm in ('stock','fast'):
   f=OUT/arm/'outputs'/(case['id']+'.npz');m=json.loads(f.with_suffix('.json').read_text())
   assert sha(f)==m['sha256'] and m['input_sha256']==sha(input_path) and m['protocol_sha256']==sha(OUT/'protocol.json')
   q=np.load(f)['flow'][0,1][:,mask].mean(1);assert np.isfinite(q).all()
   background=q[11:21].mean(0);values[arm+'_raw']=q;values[arm+'_subtract']=q-background
  data.append(dict(case=case,values=values,truth=positions(case)[1].astype(float)))
 gains={}
 for name in ('stock_raw','stock_subtract','fast_raw','fast_subtract'):
  X=np.concatenate([x['values'][name][21:79].ravel() for x in data if x['case']['split']=='calibration'])
  Y=np.concatenate([x['truth'][21:79].ravel() for x in data if x['case']['split']=='calibration'])
  denom=float(X@X);assert denom>0;g=float(X@Y/denom);assert g>0;gains[name]=g
 rows=[]
 for x in data:
  c=x['case'];truth=x['truth'];results={}
  for name,q in x['values'].items():
   for variant,g in [('uncalibrated',1.),('calibrated',gains[name])]:
    out=q*g;delta=out[21:79]-truth[21:79];axis=c['axis'];after=out[55:79,axis]
    correct=after<0; hit=next((i for i in range(len(correct)-2) if correct[i:i+3].all()),None)
    results[name+'_'+variant]=dict(epe=float(np.linalg.norm(delta,axis=1).mean()),
      before_epe=float(np.linalg.norm(out[21:55]-truth[21:55],axis=1).mean()),
      after_epe=float(np.linalg.norm(out[55:79]-truth[55:79],axis=1).mean()),
      latency=hit,correct_before_fraction=float((out[21:55,axis]>0).mean()),
      initial_response_norm=float(np.linalg.norm(out[21:31],axis=1).mean()))
  results['zero']=dict(epe=float(np.linalg.norm(truth[21:79],axis=1).mean()),latency=None)
  rows.append(dict(case=c,results=results))
 groups={}
 for split in ('calibration','evaluation'):
  for start in ('still','moving'):
   for contrast in (1.,.25):
    cohort=[x for x in rows if x['case']['split']==split and x['case']['start']==start and x['case']['contrast']==contrast];assert len(cohort)==16
    sub={}
    for name in cohort[0]['results']:
     vals=[x['results'][name] for x in cohort];entry={'epe':float(np.mean([z['epe'] for z in vals]))}
     if name!='zero':
      entry.update(before_epe=float(np.mean([z['before_epe'] for z in vals])),after_epe=float(np.mean([z['after_epe'] for z in vals])),
       latency_identified=sum(z['latency'] is not None for z in vals),
       latency_mean=float(np.mean([z['latency'] for z in vals])) if all(z['latency'] is not None for z in vals) else None,
       correct_before_fraction=float(np.mean([z['correct_before_fraction'] for z in vals])),
       initial_response_norm=float(np.mean([z['initial_response_norm'] for z in vals])))
     sub[name]=entry
    groups[f'{split}_{start}_{contrast}']=sub
 a=groups['evaluation_still_1.0'];b=groups['evaluation_still_0.25'];c=groups['evaluation_moving_1.0'];d=groups['evaluation_moving_0.25']
 check={}
 for label,g in [('still_normal',a),('still_low',b),('moving_normal',c),('moving_low',d)]:
  f=g['fast_subtract_calibrated'];raw=g['fast_raw_calibrated'];stock=g['stock_subtract_calibrated']
  check[label+'_offset_epe']=f['epe']<=raw['epe']
  check[label+'_vs_stock_epe']=f['epe']<=stock['epe']
  check[label+'_latency_complete']=f['latency_mean'] is not None and stock['latency_mean'] is not None
  check[label+'_latency_no_worse']=bool(check[label+'_latency_complete'] and f['latency_mean']<=stock['latency_mean'])
 result=dict(complete=True,exploratory_gate=all(check.values()),checks=check,gains=gains,groups=groups,rows=rows,
   protocol_sha256=sha(OUT/'protocol.json'),limits='Четыре калибровочных и четыре отложенных исходных TRAIN-видео; синтетический глобальный сдвиг, одна модель; неподвижность при оценке смещения не известна алгоритму; нет HOTA или реальных объектов.')
 (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps(dict(gains=gains,checks=check,groups={k:v for k,v in groups.items() if k.startswith('evaluation')}),ensure_ascii=False))
if __name__=='__main__':run()
