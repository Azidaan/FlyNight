import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(OUT/'matched_query_result.json');p=read(OUT/'matched_query_protocol.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(np.array(g['points']));assert sha(OUT/'matched_query_samples.npz')==r['samples_sha256'];maxerr=0.;interp=0.;checked=0
def inside(x,b):return all(b[i]<=x[i]<=b[i+2] for i in [0,1])
def score(a,b,geom,arm,head):
 if len(a)<3:return None
 active=[max(a[:,j])!=min(a[:,j]) for j in range(34)];activeb=[max(b[:,j])!=min(b[:,j]) for j in range(34)]
 if not any(active) or not any(activeb):return None
 channels=[]
 for j in range(34):
  ma=math.fsum(a[:,j])/len(a);mb=math.fsum(b[:,j])/len(b);va=math.fsum((float(x)-ma)**2 for x in a[:,j])/len(a) if active[j] else 0.;vb=math.fsum((float(x)-mb)**2 for x in b[:,j])/len(b) if activeb[j] else 0.;cov=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a[:,j],b[:,j]))/len(a) if active[j] and activeb[j] else 0.;den=math.sqrt(va*vb) if head=='native' else va+vb+(ma-mb)**2;channels.append((cov if head=='native' else 2*cov)/den*34/sum(active) if den and active[j] else 0.)
 h=r['heads'][arm][head];features=[(geom[i]-h['geometry']['mean'][i])/h['geometry']['scale'][i] for i in range(len(geom))]+[(channels[i]-h['signal_mean'][i])/h['signal_scale'][i] for i in range(34)];return math.fsum(float(x)*float(y) for x,y in zip(features,h['theta'][:-1]))+h['theta'][-1]
with np.load(OUT/'matched_query_samples.npz') as s:
 for meta in r['meta']:
  key=f"{meta['case']}_{meta['representation']}";xy=s[key+'_reference_world'];queries=[s[key+'_'+mode+'_query'] for mode in ['original','detector_zero','gt_affine_zero','gt_center_zero']];crop=meta['current_crop'];valid=[mesh.find_simplex(q)>=0 for q in queries];world=[q/float(g['eye_size'])*(np.array(crop[2:])-crop[:2])+crop[:2] for q in queries];common=np.array([inside(x,meta['reference_gt']) and all(v[i] for v in valid) for i,x in enumerate(xy)]);dual=np.array([common[i] and all(inside(w[i],meta['current_gt']) for w in world) for i in range(len(xy))]);assert np.array_equal(common,s[key+'_common_referenceGT']);assert np.array_equal(dual,s[key+'_common_dualGT'])
  for row in [x for x in r['records'] if x['case']==meta['case'] and x['representation']==meta['representation']]:
   mask=common if row['support']=='common_referenceGT' else dual;a=s[key+'_a_full'][mask];b=s[key+'_'+row['support']+'_'+row['mode']+'_b'];q=s[key+'_'+row['mode']+'_query'][mask];other=LinearNDInterpolator(mesh,s[key+'_map_current'])(q);interp=max(interp,float(abs(b-other).max(initial=0)));value=score(a,b,meta['geometry'],meta['representation'],row['head']);assert (value is None)==(row['score'] is None)
   if value is not None:maxerr=max(maxerr,abs(value-row['score']))
   checked+=1
assert maxerr<1e-9 and interp<1e-9,(maxerr,interp)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for n,h in p['sources'].items():assert sha(OUT/n)==h,n
with (OUT/'matched_query_audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,checked_scores=checked,max_scalar_score_error=maxerr,max_scipy_interpolation_error=interp,shared='Original maps/heads/GT labels and stored queries; masks reconstructed independently and interpolation rerun through SciPy.'),f,indent=2)
