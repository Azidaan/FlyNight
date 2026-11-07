"""Recompute sufficient statistics, per-case outputs, phase and summary from saved data."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_internal_signal_probe_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json') and p['source_protocol_sha256']==sha(SRC/'protocol.json');cases=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'))['cases'];mask=np.load(SRC/'geometry.npz')['mask'];cv2.setNumThreads(2);items=[];start=time.perf_counter()
 for c in cases:
  inp=SRC/'inputs'/(c['id']+'.npz');im=json.loads(inp.with_suffix('.json').read_text());assert sha(inp)==im['sha256'];f=np.load(inp)['frames'];a={}
  for arm in ('stock','fast'):
   q=SRC/arm/'outputs'/(c['id']+'.npz');m=json.loads(q.with_suffix('.json').read_text());assert sha(q)==m['sha256'] and m['input_sha256']==im['sha256'] and m['protocol_sha256']==p['source_protocol_sha256'];z=np.load(q);act=z['activity'][0];flow=z['flow'][0,1][:,mask].mean(axis=1);a[arm]=dict(activity=act-act[0],decoder=flow-flow[0])
  items.append(dict(case=c,features=a,truth=positions(c)[1].astype(float),frames=f))
 def head(x,arm,name):return x['features'][arm]['decoder'] if name=='decoder' else x['features'][arm]['activity'][:,p['heads'][name]]
 weights={}; maxdiff=0.;valid=True
 for arm in ('stock','fast'):
  weights[arm]={}
  for name in p['heads']:
   d=2 if name=='decoder' else len(p['heads'][name]);xx=np.zeros((d,d));xy=np.zeros((d,2))
   for x in items:
    if x['case']['split']=='calibration':q=head(x,arm,name)[31:79];xx+=q.T@q;xy+=q.T@x['truth'][31:79]
   ridge=float(np.trace(xx)/d*1e-4);w=np.linalg.solve(xx+np.eye(d)*ridge,xy) if ridge>0 else np.zeros((d,2));weights[arm][name]=w;original=np.asarray(r['weights'][arm][name]['coefficients']);valid &= bool(np.allclose(w,original,rtol=1e-7,atol=1e-6))
 for x,row in zip(items,r['rows']):
  assert x['case']==row['case'];y=x['truth'];c=x['case']
  for arm in ('stock','fast'):
   for name in p['heads']:
    q=head(x,arm,name)@weights[arm][name];after=float(np.linalg.norm(q[55:79]-y[55:79],axis=1).mean());before=float(np.linalg.norm(q[31:55]-y[31:55],axis=1).mean());sign=q[55:79,c['axis']]<0;hit=next((j for j in range(22) if sign[j:j+3].all()),None);ref=row['results'][arm+'_'+name];maxdiff=max(maxdiff,abs(after-ref['after_epe']),abs(before-ref['before_epe']),abs(float(sign.mean())-ref['after_correct']));valid &= hit==ref['latency']
  if c['split']=='evaluation':
   phase=[];invalid=0
   for t in range(55,79):
    a=x['frames'][t-1].astype(np.float32);b=x['frames'][t].astype(np.float32)
    if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));invalid+=1;continue
    v,_=cv2.phaseCorrelate(a,b)
    if not np.isfinite(v).all():phase.append((0.,0.));invalid+=1
    else:phase.append(v)
   score=float(np.linalg.norm(np.asarray(phase)-y[55:79],axis=1).mean());maxdiff=max(maxdiff,abs(score-row['results']['phase']['after_epe']));valid &= invalid==row['results']['phase']['invalid']
 for split in ('calibration','evaluation'):
  for st in ('still','moving'):
   for ct in (1.,.25):
    cohort=[z for z in r['rows'] if z['case']['split']==split and z['case']['start']==st and z['case']['contrast']==ct];group=r['groups'][f'{split}_{st}_{ct}'];assert len(cohort)==16
    for name in group:maxdiff=max(maxdiff,abs(float(np.mean([z['results'][name]['after_epe'] for z in cohort]))-group[name]['after_epe']))
 result=dict(passed=bool(valid and maxdiff<1e-7),all_input_output_hashes_verified=True,heads_refit_independently=bool(valid),max_metric_difference=maxdiff,report_sha256=sha(OUT/'report.json'),scope='Independent sufficient-statistics and metrics on saved outputs; shared source geometry/GT/phase implementation; no new network simulation or dataset.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));assert result['passed']
if __name__=='__main__':main()
