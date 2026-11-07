"""Independently recalculate dynamic static-reference outputs and metrics."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_counterfactual_static_recovery1_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json') and p['source_protocol_sha256']==sha(SRC/'protocol.json');old=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));mask=np.load(SRC/'geometry.npz')['mask'];cv2.setNumThreads(2);start=time.perf_counter();refs={};items=[];valid=True
 def get(c):
  f=SRC/'inputs'/(c['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256'];images=np.load(f)['frames'];v={}
  for arm in ('stock','fast'):
   path=SRC/arm/'outputs'/(c['id']+'.npz');meta2=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta2['sha256'] and meta2['input_sha256']==sha(f) and meta2['protocol_sha256']==p['source_protocol_sha256'];v[arm]=np.load(path)['flow'][0,1][:,mask].mean(1)
  return images,v
 for c in old['cases']:
  key=(c['scene'],c['contrast'],c['speed'],c['axis'])
  if c['start']=='still':refs[key]=get(c)
 for c in old['cases']:
  key=(c['scene'],c['contrast'],c['speed'],c['axis']);rf,rv=refs[key];assert all(np.array_equal(rf[0],f) for f in rf[:31]);frames,v=(rf,rv) if c['start']=='still' else get(c);assert np.array_equal(frames[0],rf[0]);truth=positions(c)[1][1:31].astype(float);values={}
  for arm in ('stock','fast'):
   values[arm+'_raw']=v[arm][1:31];values[arm+'_first']=v[arm][1:31]-v[arm][0];values[arm+'_counter']=v[arm][1:31]-rv[arm][1:31]
  items.append(dict(case=c,truth=truth,values=values,frames=frames))
 diff=0.;gains={}
 for name in items[0]['values']:
  cross=energy=0.
  for i in items:
   if i['case']['split']=='calibration':cross+=float(np.sum(i['values'][name]*i['truth']));energy+=float(np.sum(i['values'][name]**2))
  g=cross/energy;gains[name]=g if g>0 else None; valid &= (gains[name] is None)==(r['gains'][name] is None); diff=max(diff,abs(g-r['gains'][name])) if gains[name] is not None else diff
 for i,row in zip(items,r['rows']):
  assert i['case']==row['case']
  for name,values in i['values'].items():
   gain=gains[name]; raw=float(np.linalg.norm(values-i['truth'],axis=1).mean());diff=max(diff,abs(raw-row['metrics'][name]['uncalibrated_epe']));valid &= (gain is None)==(row['metrics'][name]['epe'] is None);got=float(np.linalg.norm(values*gain-i['truth'],axis=1).mean()) if gain is not None else None;diff=max(diff,abs(got-row['metrics'][name]['epe'])) if got is not None else diff
  z=float(np.linalg.norm(i['truth'],axis=1).mean());diff=max(diff,abs(z-row['metrics']['zero']['epe']))
  if i['case']['split']=='evaluation':
   phase=[];bad=0
   for t in range(1,31):
    a=i['frames'][t-1].astype(np.float32);b=i['frames'][t].astype(np.float32)
    if np.ptp(a)==0 or np.ptp(b)==0:phase.append((0.,0.));bad+=1;continue
    xy,_=cv2.phaseCorrelate(a,b)
    if not np.isfinite(xy).all():phase.append((0.,0.));bad+=1
    else:phase.append(xy)
   v=float(np.linalg.norm(np.asarray(phase)-i['truth'],axis=1).mean());diff=max(diff,abs(v-row['metrics']['phase']['epe']));valid &= bad==row['metrics']['phase']['invalid']
 for split in ('calibration','evaluation'):
  for startkind in ('still','moving'):
   for contrast in (1.,.25):
    cohort=[z for z in r['rows'] if z['case']['split']==split and z['case']['start']==startkind and z['case']['contrast']==contrast];assert len(cohort)==16;group=r['groups'][f'{split}_{startkind}_{contrast}']
    for name in group:
     if group[name]['epe'] is not None:diff=max(diff,abs(float(np.mean([x['metrics'][name]['epe'] for x in cohort]))-group[name]['epe']))
     if group[name]['uncalibrated_epe'] is not None:diff=max(diff,abs(float(np.mean([x['metrics'][name]['uncalibrated_epe'] for x in cohort]))-group[name]['uncalibrated_epe']))
 res=dict(passed=bool(valid and diff<1e-10),all_input_and_output_sha_and_static_pairs_verified=True,max_numeric_difference=diff,report_sha256=sha(OUT/'report.json'),scope='Independent gain and EPE recomputation on saved FP64 outputs; shared source geometry/ground truth and OpenCV phase; no repeat of FlyVis or independently collected data.',seconds=time.perf_counter()-start);(OUT/'independent_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(res));assert res['passed']
if __name__=='__main__':main()
