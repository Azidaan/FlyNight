"""Independent evaluator of saved paired decoder outputs; cluster by source scene."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_t4t5_paired_eval_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['evaluator_sha256'];done=json.loads((OUT/'inference_completed.json').read_text(encoding='utf-8'));assert done['complete'] and done['cases']==len(p['cases']) and done['protocol_sha256']==sha(OUT/'protocol.json');cv2.setNumThreads(2);mask=np.load(SRC/'geometry.npz')['mask'];rows=[];start=time.perf_counter()
 for c in p['cases']:
  target=OUT/'cases'/(c['id']+'.npz');meta=json.loads(target.with_suffix('.json').read_text());ref=SRC/'stock'/'outputs'/(c['id']+'.npz');inp=SRC/'inputs'/(c['id']+'.npz');assert sha(target)==meta['sha256'] and sha(ref)==meta['reference_sha256'] and sha(inp)==meta['input_sha256'] and meta['protocol_sha256']==sha(OUT/'protocol.json') and meta['parameter_sha']==done['parameter_sha'];z=np.load(target);original=np.load(ref)['flow'][0,1];assert set(z.files)==set(p['arms']) and all(z[k].shape==(79,721,2) and np.isfinite(z[k]).all() for k in z.files);assert np.max(np.abs(z['reference']-original))<=1e-5 and np.array_equal(z['reference'],z['identity_t4']) and meta['source_unchanged'] and meta['nonselected_exact'];truth=positions(c)[1].astype(float);refmean=z['reference'][:,mask].mean(1);metrics={}
  for name in p['arms']:
   y=z[name][:,mask].mean(1);sign=y[55:79,c['axis']]<0;hit=next((i for i in range(22) if sign[i:i+3].all()),None);metrics[name]=dict(epe=float(np.linalg.norm(y[55:79]-truth[55:79],axis=1).mean()),sign_fraction=float(sign.mean()),latency=hit,censored_latency=24 if hit is None else hit,mean_shift=float(np.linalg.norm((y-refmean)[1:79],axis=1).mean()))
  frames=np.load(inp)['frames'];phase=[];invalid=0
  for t in range(55,79):
   a=frames[t-1].astype(np.float32);b=frames[t].astype(np.float32)
   if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));invalid+=1;continue
   v,_=cv2.phaseCorrelate(a,b)
   if not np.isfinite(v).all():phase.append((0.,0.));invalid+=1
   else:phase.append(v)
  metrics['phase']=dict(epe=float(np.linalg.norm(np.asarray(phase)-truth[55:79],axis=1).mean()),invalid=invalid);metrics['zero']=dict(epe=float(np.linalg.norm(truth[55:79],axis=1).mean()));rows.append(dict(case=c,metrics=metrics,output_sha256=meta['sha256']))
 groups={};paired={};checks={};rng=np.random.default_rng(p['bootstrap_seed']);draw=rng.integers(0,8,size=(p['bootstrap_samples'],8));lo=(1-p['bootstrap_ci'])/2;hi=1-lo
 for ct in (1.,.25):
  cohort=[z for z in rows if z['case']['contrast']==ct];assert len(cohort)==32;g={}
  for arm in list(p['arms'])+['phase','zero']:
   a=[z['metrics'][arm] for z in cohort];v=dict(epe=float(np.mean([x['epe'] for x in a])))
   if arm in p['arms']:v.update(censored_latency=float(np.mean([x['censored_latency'] for x in a])),not_found=sum(x['latency'] is None for x in a),sign_fraction=float(np.mean([x['sign_fraction'] for x in a])),mean_shift=float(np.mean([x['mean_shift'] for x in a])))
   if arm=='phase':v['invalid']=sum(x['invalid'] for x in a)
   g[arm]=v
  groups[str(ct)]=g
  for control in ('reference','control_other'):
   ds=np.array([np.mean([z['metrics']['scale_t4']['epe']-z['metrics'][control]['epe'] for z in cohort if z['case']['scene']==scene]) for scene in range(8)]);samples=ds[draw].mean(1);interval=np.quantile(samples,[lo,hi]);paired[f'{ct}_{control}']=dict(mean=float(ds.mean()),lower=float(interval[0]),upper=float(interval[1]),scene_differences=ds.tolist());checks[f'{ct}_{control}_epe_upper_negative']=bool(interval[1]<0);checks[f'{ct}_{control}_latency_no_worse']=g['scale_t4']['censored_latency']<=g[control]['censored_latency'] and g['scale_t4']['not_found']<=g[control]['not_found']
 result=dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),inference_completed_sha256=sha(OUT/'inference_completed.json'),cases=len(rows),groups=groups,paired=paired,checks=checks,primary_met=all(checks.values()),rows=rows,scope=p['limits'],seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(dict(groups=groups,paired=paired,checks=checks,primary_met=result['primary_met']),ensure_ascii=False))
if __name__=='__main__':main()
