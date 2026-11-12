import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'proposed_protocol.json');r=read(OUT/'result.json');assert sha(OUT/'samples.npz')==r['samples_sha256'];maxerr=0.;checked=0
def scalar_inside(x,bb):return bb['x1']<=x[0]<=bb['x2'] and bb['y1']<=x[1]<=bb['y2']
def score(a,b,g,arm,head):
 if len(a)<3:return None
 active=np.max(a,0)!=np.min(a,0);activeb=np.max(b,0)!=np.min(b,0)
 if not active.any() or not activeb.any():return None
 # Independent moments via E[x*y]-E[x]*E[y], stable centered dot for covariance.
 am=np.mean(a,0);bm=np.mean(b,0);ac=a-am;bc=b-bm;ac[:,~active]=0.;bc[:,~activeb]=0.;cov=np.einsum('ij,ij->j',ac,bc)/len(a);va=np.einsum('ij,ij->j',ac,ac)/len(a);vb=np.einsum('ij,ij->j',bc,bc)/len(b)
 den=np.sqrt(va*vb) if head=='native' else va+vb+(am-bm)**2
 q=np.divide(cov if head=='native' else 2*cov,den,out=np.zeros(34),where=den>0);q[~active]=0;q*=34/active.sum();h=r['heads'][arm][head];geom=[(g[i]-h['geometry']['mean'][i])/h['geometry']['scale'][i] for i in range(len(g))];features=geom+[(q[i]-h['signal_mean'][i])/h['signal_scale'][i] for i in range(34)];return math.fsum(float(x)*float(y) for x,y in zip(features,h['theta'][:-1]))+h['theta'][-1]
with np.load(OUT/'samples.npz') as s:
 for meta in r['meta']:
  ci=meta['case'];objects=meta['objects'];ids=meta['ids'];owners=meta['owners']
  for arm in p['representations']:
   key=f'{ci}_{arm}';a=s[key+'_a'];b=s[key+'_b'];xy=[s[key+'_reference_world'],s[key+'_current_world']]
   dual=np.array([all(scalar_inside(xy[e][i],owners[e]['box2d']) for e in [0,1]) for i in range(len(a))]);exclusive=np.array([bool(dual[i]) and all(not any(str(o['id'])!=ids[e] and 'box2d' in o and scalar_inside(xy[e][i],o['box2d']) for o in objects[e]) for e in [0,1]) for i in range(len(a))]);assert np.array_equal(dual,s[key+'_dual_gt']);assert np.array_equal(exclusive,s[key+'_dual_exclusive'])
   masks={'full':np.ones(len(a),bool),'identity':np.ones(len(a),bool),'dual_gt':dual,'dual_exclusive':exclusive}
   for row in [z for z in r['records'] if z['case']==ci and z['representation']==arm]:
    mask=masks[row['mode']];z=score(a[mask],b[mask],meta['geometry'],arm,row['head']);assert (z is None)==(row['score'] is None)
    if z is not None:maxerr=max(maxerr,abs(z-row['score']))
    checked+=1
   for control in [z for z in r['controls'] if z['case']==ci and z['representation']==arm]:
    ix=s[key+'_'+control['mode']+'_control_indices']
    for indices,expected in zip(ix,control['scores']):
     assert len(np.unique(indices))==len(indices);z=score(a[indices],b[indices],meta['geometry'],arm,control['head']);assert (z is None)==(expected is None)
     if z is not None:maxerr=max(maxerr,abs(z-expected))
     checked+=1
 for row in r['oracle_records']:
  k=row['sample_key'];meta=next(m for m in r['meta'] if m['case']==row['case']);z=score(s[k+'_a'],s[k+'_b'],meta['geometry'],row['representation'],row['head']);assert (z is None)==(row['score'] is None)
  if z is not None:maxerr=max(maxerr,abs(z-row['score']))
  checked+=1
assert maxerr<1e-9,maxerr
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for n,h in p['source_hashes'].items():assert sha(OUT/n)==h,n
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,checked_scores=checked,max_absolute_error=maxerr,mask_reconstruction='independent scalar point-in-box; annotations saved by worker share original labels and anchor routing',score_reconstruction='independent einsum moments and math.fsum linear head; no tracking/readout code imports',limitations='Shared archived maps, frozen heads, GT labels and coordinate generator. Pixelownership not proven.'),f,indent=2)
print('AUDIT_PASS',checked,maxerr)
