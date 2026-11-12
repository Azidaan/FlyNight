from common import *
import numpy as np
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
p=verify();source=read(ROOT/'research_candidate_support_probe_20261112/scores.json');heads=source['heads'];rng=np.random.default_rng(20261112)
with np.load(ROOT/'research_candidate_support_probe_20261112/patches.npz',allow_pickle=False) as z:patches={k:z[k] for k in z.files}
def score(a,b,meta,head):
 if head=='agreement':q=agreement_channels(a,b,mean_penalty=True)
 else:
  x,y,an,bn,active,active_b=_center_pair(a,b);assert active.any() and active_b.any();q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
 assert q is not None;h=heads[meta['representation']][head];g=(np.array(meta['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];features=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta']);return dict(score=float(features@theta[:-1]+theta[-1]),channels=q)
records=[];parts={};decomp=[];controls=[]
for meta in source['meta']:
 key=str(meta['case'])+'_'+meta['representation'];origA=patches[key+'_a'];origB=patches[key+'_b']
 for support in p['parameters']['supports']:
  mask=np.ones(len(origA),bool) if support=='full' else patches[key+'_'+support];a=origA[mask];b=origB[mask];xy=patches[key+'_world'][mask];X=xy-xy.mean(0);assert np.linalg.matrix_rank(X)==2;Q=np.linalg.qr(X)[0];ma=a.mean(0);mb=b.mean(0);ac=a-ma;bc=b-mb;ta=Q@(Q.T@ac);tb=Q@(Q.T@bc);ra=ac-ta;rb=bc-tb
  k=key+'_'+support
  for name,value in dict(a=a,b=b,xy=xy,ta=ta,tb=tb,ra=ra,rb=rb).items():parts[k+'_'+name]=value
  decomp.append(dict(key=k,meta=meta,support=support,cov_full=(ac*bc).mean(0),cov_trend=(ta*tb).mean(0),cov_residual=(ra*rb).mean(0),var_a=(ac*ac).mean(0),var_trend_a=(ta*ta).mean(0),var_residual_a=(ra*ra).mean(0),var_b=(bc*bc).mean(0),var_trend_b=(tb*tb).mean(0),var_residual_b=(rb*rb).mean(0)))
  for mode,A,B in [('identity',a,b),('residual',ra+ma,rb+mb),('trend_only',ta+ma,tb+mb)]:
   for head in ['native','agreement']:
    z=score(A,B,meta,head)
    if mode=='identity':
     old=next(x for x in source['records'] if x['old_id']==meta['old_id'] and x['representation']==meta['representation'] and x['head']==head and x['mode']==('full' if support=='full' else support));assert abs(z['score']-old['score'])<1e-9
    records.append(dict(key=k,old_id=meta['old_id'],representation=meta['representation'],head=head,support=support,mode=mode,count=len(a),accepted=z['score']>0,**z))
  for control in range(16):
   G=rng.normal(size=(len(a),2));G-=G.mean(0);R=np.linalg.qr(G)[0];A=a-R@(R.T@ac);B=b-R@(R.T@bc)
   for head in ['native','agreement']:controls.append(dict(key=k,old_id=meta['old_id'],representation=meta['representation'],support=support,head=head,control=control,**score(A,B,meta,head)))
verify()
with (OUT/'parts.npz').open('xb') as f:np.savez_compressed(f,**parts)
save('result.json',dict(passed=True,records=records,controls=controls,decomposition=decomp,heads=heads,parts_sha256=sha(OUT/'parts.npz'),core_calls=0,new_fits=0,tracker_calls=0))
print(json.dumps([x for x in records if x['representation']=='C34' and x['support']=='detector_and_annotation' and x['mode']!='trend_only'],default=cast))
