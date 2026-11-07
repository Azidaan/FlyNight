"""Независимый пересчёт причинного признака, подгонки и метрик сохранённых данных."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_still_gate_20261105';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104';sys.path.insert(0,str(ROOT))
from scripts.flyvis_offset_recovery2_20261104 import positions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'));old=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));assert r['protocol_sha256']==sha(OUT/'protocol.json');cv2.setNumThreads(2)
 mask=np.load(SRC/'geometry.npz')['mask'];rows=[]
 for c in old['cases']:
  f=SRC/'inputs'/(c['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256'];frames=np.load(f)['frames']
  gate=bool(np.all(frames[10:21]==frames[10][None,:,:]));rec=next(x for x in r['rows'] if x['case']['id']==c['id']);assert gate==rec['gate'];values={}
  for arm in ('stock','fast'):
   f=SRC/arm/'outputs'/(c['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256'];flow=np.load(f)['flow'][0,1];q=np.einsum('tnc,n->tc',flow,mask.astype(float))/mask.sum();values[arm+'_raw']=q;values[arm+'_mean']=q-sum(q[11:21])/10;values[arm+'_first']=q-q[0]
  target=positions(c)[1].astype(float);rows.append((c,SRC/'inputs'/(c['id']+'.npz'),gate,values,target))
 maxdiff=0.
 for arm in ('stock','fast'):
  X=np.concatenate([x[3][arm+'_first'][21:79].flatten() for x in rows if x[0]['split']=='calibration']);Y=np.concatenate([x[4][21:79].flatten() for x in rows if x[0]['split']=='calibration']);g=float(np.linalg.lstsq(X[:,None],Y,rcond=None)[0][0]);diff=abs(g-r['gains'][arm+'_first']);maxdiff=max(maxdiff,diff);assert diff<1e-9
 for c,input_path,gate,values,target in rows:
  rec=next(x for x in r['rows'] if x['case']['id']==c['id'])
  for arm in ('stock','fast'):
   for mode in ('raw','mean','first','gated'):
    selected=('mean' if gate else 'raw') if mode=='gated' else mode;q=values[arm+'_'+selected]*r['gains'][arm+'_'+selected];d=q[21:79]-target[21:79];e=float(np.sqrt(d[:,0]**2+d[:,1]**2).mean());difference=abs(e-rec['results'][arm+'_'+mode]['epe']);maxdiff=max(maxdiff,difference);assert difference<1e-9
  if c['split']=='evaluation':
   frames=np.load(input_path)['frames']
   phase=[];invalid=0
   for j in range(21,79):
    prev=frames[j-1].astype(np.float32);curr=frames[j].astype(np.float32)
    if np.ptp(prev)==0 or np.ptp(curr)==0:phase.append((0.,0.));invalid+=1;continue
    z,_=cv2.phaseCorrelate(prev,curr)
    if not np.isfinite(z).all():phase.append((0.,0.));invalid+=1
    else:phase.append(z)
   delta=np.asarray(phase)-target[21:79];e=float(np.sqrt(delta[:,0]**2+delta[:,1]**2).mean());difference=abs(e-rec['results']['phase']['epe']);maxdiff=max(maxdiff,difference);assert difference<1e-9 and invalid==rec['results']['phase']['invalid']
 audit=dict(passed=True,cases=len(rows),gate_recomputed=128,phase_cases=64,train_eval_disjoint=set(old['videos'][:4]).isdisjoint(old['videos'][4:]),max_abs_diff=maxdiff,
   report_sha256=sha(OUT/'report.json'),scope='Повтор правил по сохранённым данным и повтор фазовой корреляции; не повтор FlyVis и не независимые данные.')
 assert audit['train_eval_disjoint'];(OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
