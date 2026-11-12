from common import *
import numpy as np,math
p=verify();old=read(ROOT/'research_spatial_trend_probe_20261112/result.json')
with np.load(ROOT/'research_spatial_trend_probe_20261112/parts.npz') as z:parts={k:z[k] for k in z.files}
rows=[];maximum=0.
for d in old['decomposition']:
 k=d['key'];A=parts[k+'_ra']+parts[k+'_a'].mean(0);B=parts[k+'_rb']+parts[k+'_b'].mean(0);ma=A.mean(0);mb=B.mean(0);a=A-ma;b=B-mb;va=(a*a).mean(0);vb=(b*b).mean(0);active=np.ptp(A,axis=0)!=0;activeb=np.ptp(B,axis=0)!=0
 for head in ['native','agreement']:
  h=old['heads'][d['meta']['representation']][head];theta=np.array(h['theta']);g=(np.array(d['meta']['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];offset=float(g@theta[:7]+theta[-1]+(-np.array(h['signal_mean'])/h['signal_scale'])@theta[7:-1]);den=np.sqrt(va*vb) if head=='native' else (va+vb+(ma-mb)**2)/2;coef=np.divide(theta[7:-1]/h['signal_scale'],den,out=np.zeros(34),where=(den>0)&active&activeb)*34/active.sum();scores=[]
  for shift in range(len(A)):
   C=np.roll(B,shift,axis=0);c=C-C.mean(0);cov=(a*c).mean(0);q=np.divide(cov,den,out=np.zeros(34),where=(den>0)&active&activeb)*34/active.sum();features=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];score=float(features@theta[:-1]+theta[-1]);ind=offset+math.fsum(float(w)*math.fsum(float(x)*float(y) for x,y in zip(a[:,j],c[:,j]))/len(A) for j,w in enumerate(coef));maximum=max(maximum,abs(score-ind));scores.append(score)
  expected=next(x['score'] for x in old['records'] if x['key']==k and x['head']==head and x['mode']=='residual');assert abs(scores[0]-expected)<1e-10
  avg=math.fsum(scores)/len(scores);maximum=max(maximum,abs(avg-offset));rows.append(dict(old_id=d['meta']['old_id'],representation=d['meta']['representation'],support=d['support'],head=head,count=len(A),identity_score=scores[0],zero_covariance_score=offset,aligned_covariance_contribution=scores[0]-offset,all_scores=scores,average=avg,min=min(scores),max=max(scores),accepted_count=sum(x>0 for x in scores)))
assert maximum<1e-10,maximum;verify();save('result.json',dict(passed=True,max_error=maximum,records=rows,total_permutation_scores=sum(len(x['all_scores']) for x in rows),baseline_scores=len(rows),inputs_unchanged=True,core_calls=0,new_fits=0,tracker_calls=0))
print(json.dumps([{k:v for k,v in x.items() if k not in ['all_scores']} for x in rows if x['old_id'] in [574,232,832] and x['representation']=='C34' and x['support']=='detector_and_annotation']))
