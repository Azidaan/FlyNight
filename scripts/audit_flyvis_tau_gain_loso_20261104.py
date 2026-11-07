"""Повтор коэффициентов и показателей альтернативным способом по исходным полям."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_tau_gain_loso_20261104'
STOCK=ROOT/'runs/flyvis_y1_latency_fp64_20261104';FAST=ROOT/'runs/flyvis_tau_half_20261104'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
 assert r['protocol_sha256']==sha(OUT/'protocol.json')
 mask=np.load(FAST/'geometry.npz')['mask'];independent=[]
 for case in p['cases']:
  trace={}
  for name,folder in [('stock',STOCK),('fast',FAST)]:
   f=folder/'outputs'/(case['id']+'.npz');meta=json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
   assert sha(f)==meta['sha256']
   v=np.load(f)['flow'][0,1];trace[name]=np.einsum('tnc,n->tc',v,mask.astype(float))/mask.sum()
  y=np.zeros((79,2));y[31:55,case['axis']]=case['speed']
  if case['event']=='reverse':y[55:79,case['axis']]=-case['speed']
  independent.append((case,trace,y))
 maxdiff=0.
 for fold in range(4):
  for arm in ['stock','fast']:
   X=np.concatenate([x[1][arm][31:79].ravel() for x in independent if x[0]['scene']!=fold]);Y=np.concatenate([x[2][31:79].ravel() for x in independent if x[0]['scene']!=fold])
   k=float(np.linalg.lstsq(X[:,None],Y,rcond=None)[0][0]);delta=abs(k-r['gains'][str(fold)][arm]);maxdiff=max(maxdiff,delta);assert delta<1e-10
  for case,traces,y in [x for x in independent if x[0]['scene']==fold]:
   row=next(x for x in r['rows'] if x['case']['id']==case['id'])
   for arm in ['stock','fast']:
    for version,k in [('raw',1),('calibrated',r['gains'][str(fold)][arm])]:
     q=traces[arm]*k;err=q[31:79]-y[31:79]
     val=float(np.sqrt(err[:,0]**2+err[:,1]**2).mean());st=q[21:31];noise=float(np.sqrt(np.var(st,axis=0).sum()))
     for key,value in [('epe',val),('static_noise',noise)]:
      delta=abs(value-row['arms'][arm+'_'+version][key]);maxdiff=max(maxdiff,delta);assert delta<1e-10,(case['id'],arm,version,key,delta)
 audit=dict(passed=True,cases=len(independent),folds=4,coefficient_training_excludes_heldout_scene=True,
  max_abs_difference=maxdiff,report_sha256=sha(OUT/'report.json'),scope='Независимый пересчёт по сохранённым полям; сцены прежде изучались, сеть повторно не запускалась.')
 (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
