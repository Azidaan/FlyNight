"""Independently reconstruct fixed projection, fits and evaluation from saved outputs."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_equal_readout_20261105';TRAIN=ROOT/'runs/flyvis_offset_recovery2_20261104';STOCK=ROOT/'runs/flyvis_y1_latency_fp64_20261104';FAST=ROOT/'runs/flyvis_tau_half_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions as train_positions
from scripts.flyvis_y1_latency_fp64_20261104 import positions as eval_positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json');cv2.setNumThreads(2);tr=json.loads((TRAIN/'protocol.json').read_text(encoding='utf-8'))['cases'];ev=[x for x in json.loads((STOCK/'protocol.json').read_text(encoding='utf-8'))['cases'] if x['event']=='reverse'];assert p['source_train_protocol_sha256']==sha(TRAIN/'protocol.json') and p['source_stock_protocol_sha256']==sha(STOCK/'protocol.json') and p['source_fast_protocol_sha256']==sha(FAST/'protocol.json');mtr=np.load(TRAIN/'geometry.npz')['mask'];mev=np.load(STOCK/'geometry.npz')['mask'];start=time.perf_counter();items=[]
 def load(c,ist):
  base=TRAIN if ist else STOCK;ip=base/'inputs'/(c['id']+'.npz');im=json.loads(ip.with_suffix('.json').read_text());assert sha(ip)==im['sha256'];frames=np.load(ip)['frames'];v={}
  for arm in ('stock','fast'):
   root=TRAIN/arm if ist else (STOCK if arm=='stock' else FAST);q=root/'outputs'/(c['id']+'.npz');m=json.loads(q.with_suffix('.json').read_text());assert sha(q)==m['sha256'] and m['input_sha256']==im['sha256'] and m['protocol_sha256']==sha(root/'protocol.json');z=np.load(q);a=z['activity'][0];a=a-a[0];d=z['flow'][0,1][:,(mtr if ist else mev)].mean(1);d=d-d[0];projection=np.stack(((a[:,1]-a[:,3]+a[:,5]-a[:,7])/2,(a[:,2]-a[:,4]+a[:,6]-a[:,8])/2),axis=1);v[arm]={'projected':projection,'decoder':d,'all8':a[:,1:9]}
  return frames,v
 for c in tr:
  f,v=load(c,True);items.append(dict(case=c,split='train',features=v,truth=train_positions(c)[1].astype(float),frames=None))
 for c in ev:
  f,v=load(c,False);items.append(dict(case=c,split='evaluation',features=v,truth=eval_positions(c)[1].astype(float),frames=f))
 weights={};valid=True;diff=0.
 for arm in ('stock','fast'):
  weights[arm]={}
  for head in ('projected','decoder','all8'):
   d=8 if head=='all8' else 2;xx=np.zeros((d,d));xy=np.zeros((d,2))
   for x in items:
    if x['split']=='train':q=x['features'][arm][head][31:79];xx+=q.T@q;xy+=q.T@x['truth'][31:79]
   ridge=float(np.trace(xx)/d*1e-4);w=np.linalg.solve(xx+np.eye(d)*ridge,xy) if ridge>0 else np.zeros((d,2));weights[arm][head]=w;valid &= bool(np.allclose(w,np.asarray(r['weights'][arm][head]['coefficients']),rtol=1e-7,atol=1e-6))
 evalitems=[x for x in items if x['split']=='evaluation'];assert len(evalitems)==len(r['rows'])
 for x,row in zip(evalitems,r['rows']):
  assert x['case']==row['case'];c=x['case'];truth=x['truth']
  for arm in ('stock','fast'):
   for head in ('projected','decoder','all8'):
    q=x['features'][arm][head]@weights[arm][head];a=float(np.linalg.norm(q[55:79]-truth[55:79],axis=1).mean());b=float(np.linalg.norm(q[31:55]-truth[31:55],axis=1).mean());sign=q[55:79,c['axis']]<0;hit=next((t for t in range(22) if sign[t:t+3].all()),None);z=row['results'][arm+'_'+head];diff=max(diff,abs(a-z['after_epe']),abs(b-z['before_epe']),abs(float(sign.mean())-z['after_correct']));valid &= hit==z['latency']
  phase=[];invalid=0
  for t in range(55,79):
   a=x['frames'][t-1].astype(np.float32);b=x['frames'][t].astype(np.float32)
   if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));invalid+=1;continue
   xy,_=cv2.phaseCorrelate(a,b)
   if not np.isfinite(xy).all():phase.append((0.,0.));invalid+=1
   else:phase.append(xy)
  diff=max(diff,abs(float(np.linalg.norm(np.asarray(phase)-truth[55:79],axis=1).mean())-row['results']['phase']['after_epe']));valid &= invalid==row['results']['phase']['invalid']
 for ct in (1.,.25):
  cohort=[x for x in r['rows'] if x['case']['contrast']==ct];assert len(cohort)==16
  for name in r['groups'][str(ct)]:diff=max(diff,abs(float(np.mean([z['results'][name]['after_epe'] for z in cohort]))-r['groups'][str(ct)][name]['after_epe']))
 result=dict(passed=bool(valid and diff<1e-7),all_input_output_sha_verified=True,refits_agree=bool(valid),max_metric_difference=diff,report_sha256=sha(OUT/'report.json'),scope='Independent sufficient statistics and metrics, same saved signals, source truth and OpenCV phase implementation; no independent new video or rerun of FlyVis.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));assert result['passed']
if __name__=='__main__':main()
