from common import *
import numpy as np, math
p=verify();r=read(OUT/'result.json');assert sha(OUT/'parts.npz')==r['parts_sha256'];maxerr=0.;maxdec=0.;rng=np.random.default_rng(20261112)
with np.load(OUT/'parts.npz') as z:parts={k:z[k] for k in z.files}
def independent(A,B,meta,head):
 values=[];active=0
 for j in range(34):
  a=A[:,j];b=B[:,j];ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b);va=math.fsum((float(x)-ma)**2 for x in a)/len(a);vb=math.fsum((float(x)-mb)**2 for x in b)/len(b);cov=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a,b))/len(a)
  aa=np.ptp(a)!=0;bb=np.ptp(b)!=0;active+=int(aa)
  if not aa or not bb:q=0.
  elif head=='native':q=cov/math.sqrt(va*vb)
  else:
   den=va+vb+(ma-mb)**2;mse=math.fsum((float(x)-float(y))**2 for x,y in zip(a,b))/len(a);q=1-mse/den if den else 0.
  values.append(q)
 q=np.array(values)*34/active;h=r['heads'][meta['representation']][head];g=(np.array(meta['geometry'])-h['geometry']['mean'])/h['geometry']['scale'];f=np.r_[g,(q-h['signal_mean'])/h['signal_scale']];t=h['theta'];return math.fsum(float(x)*float(w) for x,w in zip(f,t[:-1]))+t[-1]
for d in r['decomposition']:
 k=d['key'];a=parts[k+'_a'];b=parts[k+'_b'];xy=parts[k+'_xy'];X=xy-xy.mean(0);ma=a.mean(0);mb=b.mean(0);ac=a-ma;bc=b-mb
 ta=X@np.linalg.solve(X.T@X,X.T@ac);tb=X@np.linalg.solve(X.T@X,X.T@bc);ra=ac-ta;rb=bc-tb
 for name,v in [('ta',ta),('tb',tb),('ra',ra),('rb',rb)]:maxdec=max(maxdec,float(np.max(abs(v-parts[k+'_'+name]))))
 for name,left,right in [('cov_full','cov_trend','cov_residual'),('var_a','var_trend_a','var_residual_a'),('var_b','var_trend_b','var_residual_b')]:maxdec=max(maxdec,float(np.max(abs(np.array(d[name])-np.array(d[left])-np.array(d[right])))))
 for mode,A,B in [('identity',a,b),('residual',ma+ra,mb+rb),('trend_only',ma+ta,mb+tb)]:
  for head in ['native','agreement']:
   old=next(x for x in r['records'] if x['key']==k and x['head']==head and x['mode']==mode);maxerr=max(maxerr,abs(independent(A,B,d['meta'],head)-old['score']))
 for control in range(16):
  G=rng.normal(size=(len(a),2));G-=G.mean(0);A=a-G@np.linalg.solve(G.T@G,G.T@ac);B=b-G@np.linalg.solve(G.T@G,G.T@bc)
  for head in ['native','agreement']:
   old=next(x for x in r['controls'] if x['key']==k and x['head']==head and x['control']==control);maxerr=max(maxerr,abs(independent(A,B,d['meta'],head)-old['score']))
assert maxerr<1e-10 and maxdec<1e-10,(maxerr,maxdec)
selected=[x for x in r['records'] if x['representation']=='C34' and x['support']=='detector_and_annotation'];gate=all(next(x for x in selected if x['old_id']==574 and x['head']==h and x['mode']=='residual')['score']<=0 and all(next(x for x in selected if x['old_id']==tid and x['head']==h and x['mode']=='residual')['score']>0 for tid in [70,1102]) for h in ['native','agreement'])
effects=[]
for head in ['native','agreement']:
 vals={x['mode']:x['score'] for x in selected if x['old_id']==574 and x['head']==head};ctrl=[x['score'] for x in r['controls'] if x['old_id']==574 and x['representation']=='C34' and x['support']=='detector_and_annotation' and x['head']==head];effects.append(dict(head=head,scores=vals,random_min=min(ctrl),random_max=max(ctrl),random_median=float(np.median(ctrl)),random_rejected=sum(x<=0 for x in ctrl)))
d=next(d for d in r['decomposition'] if d['meta']['old_id']==574 and d['meta']['representation']=='C34' and d['support']=='detector_and_annotation');variance=dict(reference_median_trend_fraction=float(np.median(np.array(d['var_trend_a'])/np.array(d['var_a']))),candidate_median_trend_fraction=float(np.median(np.array(d['var_trend_b'])/np.array(d['var_b']))))
verify();save('audit.json',dict(passed=True,baseline_scores=60,main_scores=len(r['records']),control_scores=len(r['controls']),max_independent_score_error=maxerr,max_decomposition_error=maxdec,main_gate_passed=gate,effects574=effects,variance574=variance,scope='Orthogonal decomposition and frozen-reader effects; no claim a spatial plane equals background or a particular neural mechanism.'))
print(json.dumps(dict(gate=gate,effects=effects,variance=variance,maxerr=maxerr)))
