"""Independent scalar score/interpolation audit of either registered addendum."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
prefix=sys.argv[1];assert prefix in ['source_support','factorial']
def read(p):return json.loads(p.read_text(encoding='utf8'))
p=read(OUT/(prefix+'_protocol.json'));res=read(OUT/(prefix+'_result.json'));heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom={r['old_id']:r['geometry'] for r in res['geometry']};errors=dict(interpolation=0.,channels=0.,scores=0.,source=0.)
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256((OUT/(prefix+'_samples.npz')).read_bytes()).hexdigest()==res['samples_sha256']
with np.load(OUT/(prefix+'_samples.npz')) as samples:
 for row in res['records']:
  k=row['patch_key'];A=samples[k+'_a'];B=samples[k+'_b'];q=samples[k+'_q'];ci=p['cases'].index(row['old_id']);arm=row['representation']
  if prefix=='factorial':cells=samples[f'{ci}_cells'][samples[f'{ci}_common']];m0=samples[row['map0_key']];m1=samples[row['map1_key']]
  else:cells=samples[k+'_cells'];m0=samples[f'{ci}_{arm}_map0'];m1=samples[f'{ci}_{arm}_map1']
  errors['source']=max(errors['source'],float(abs(A-m0[cells]).max()));other=LinearNDInterpolator(np.array(g['points']),m1)(q);errors['interpolation']=max(errors['interpolation'],float(abs(B-other).max()));active=sum(max(A[:,j])!=min(A[:,j]) for j in range(A.shape[1]));Q=[]
  for j in range(A.shape[1]):
   a=[float(x) for x in A[:,j]];b=[float(x) for x in B[:,j]];ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b);va=math.fsum((x-ma)**2 for x in a)/len(a);vb=math.fsum((x-mb)**2 for x in b)/len(b);cov=math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))/len(a);z=0. if max(a)==min(a) or max(b)==min(b) else cov/math.sqrt(va*vb) if row['head']=='native' else 2*cov/(va+vb+(ma-mb)**2);Q.append(z*A.shape[1]/active)
  errors['channels']=max(errors['channels'],max(abs(x-y) for x,y in zip(Q,row['channels'])))
  if row['score'] is not None:
   h=heads[arm][row['head']];features=[(x-m)/s for x,m,s in zip(geom[row['old_id']],h['geometry']['mean'],h['geometry']['scale'])]+[(x-m)/s for x,m,s in zip(Q,h['signal_mean'],h['signal_scale'])];score=math.fsum(x*y for x,y in zip(features,h['theta'][:-1]))+h['theta'][-1];errors['scores']=max(errors['scores'],abs(score-row['score']));assert (score>0)==row['accepted']
assert max(errors.values())<1e-9,errors
out=dict(passed=True,prefix=prefix,records=len(res['records']),max_errors=errors,shared='Saved source maps/queries; independent SciPy interpolation and scalar formulas, no production scoring helper imports.')
with (OUT/(prefix+'_audit.json')).open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(out)
