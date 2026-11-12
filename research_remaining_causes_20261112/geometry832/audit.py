import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
p=read(OUT/'proposed_protocol.json');result=read(OUT/'result.json');heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');metadata={x['old_id']:x for x in result['geometry']};caseids=p['cases'];errors=dict(interpolation=0.,channels=0.,score=0.,source=0.);assert hashlib.sha256((OUT/'paired_samples.npz').read_bytes()).hexdigest()==result['paired_samples_sha256']
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
with np.load(OUT/'paired_samples.npz') as data:
 for row in result['records']:
  k=row['patch_key'];A=data[k+'_a'];B=data[k+'_b'];ci=caseids.index(row['old_id']);arm=row['representation'];M=data[f'{ci}_common'] if row['support']=='common' else np.ones(len(data[f'{ci}_cells']),bool);source=data[f'{ci}_{arm}_map0'][data[f'{ci}_cells']][M];errors['source']=max(errors['source'],float(abs(A-source).max()));other=LinearNDInterpolator(np.array(geom['points']),data[f'{ci}_{arm}_map1'])(data[k+'_q']);errors['interpolation']=max(errors['interpolation'],float(abs(B-other).max()));active=sum(max(A[:,j])!=min(A[:,j]) for j in range(A.shape[1]));Q=[]
  for j in range(A.shape[1]):
   a=[float(v) for v in A[:,j]];b=[float(v) for v in B[:,j]];ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b);va=math.fsum((x-ma)**2 for x in a)/len(a);vb=math.fsum((x-mb)**2 for x in b)/len(b);cov=math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))/len(a)
   q=0. if max(a)==min(a) or max(b)==min(b) else cov/math.sqrt(va*vb) if row['head']=='native' else 2*cov/(va+vb+(ma-mb)**2);Q.append(q*A.shape[1]/active)
  errors['channels']=max(errors['channels'],max(abs(x-y) for x,y in zip(Q,row['channels'])))
  if row['score'] is not None:
   h=heads[arm][row['head']];G=metadata[row['old_id']]['geometry'];features=[(float(x)-m)/s for x,m,s in zip(G,h['geometry']['mean'],h['geometry']['scale'])]+[(x-m)/s for x,m,s in zip(Q,h['signal_mean'],h['signal_scale'])];score=math.fsum(x*y for x,y in zip(features,h['theta'][:-1]))+h['theta'][-1];errors['score']=max(errors['score'],abs(score-row['score']));assert (score>0)==row['accepted']
assert max(errors.values())<1e-9,errors
out=dict(passed=True,records=len(result['records']),max_errors=errors,shared_dependencies='Saved patch generator and source annotations shared; independent SciPy interpolation and scalar channel/head scoring do not call production scoring helpers.',limits='Checks saved patches and frozen-head arithmetic, not biological validity or physical correspondence.')
with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(out)
