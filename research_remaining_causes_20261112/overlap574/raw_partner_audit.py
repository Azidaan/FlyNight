import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,math,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(OUT/'raw_partner_result.json');assert sha(OUT/'raw_partner_samples.npz')==r['samples_sha256'];maxerr=0.;checked=0
with np.load(OUT/'raw_partner_samples.npz') as saved:
 for row in r['records']:
  k=row['sample_key'];a=saved[k+'_a'];b=saved[k+'_b']
  if len(a)<3:assert row['metrics'] is None;continue
  pear=[];con=[]
  for j in range(a.shape[1]):
   ma=math.fsum(float(x) for x in a[:,j])/len(a);mb=math.fsum(float(x) for x in b[:,j])/len(b);va=math.fsum((float(x)-ma)**2 for x in a[:,j])/len(a);vb=math.fsum((float(x)-mb)**2 for x in b[:,j])/len(b);cov=math.fsum((float(x)-ma)*(float(y)-mb) for x,y in zip(a[:,j],b[:,j]))/len(a);pear.append(cov/math.sqrt(va*vb) if va*vb>0 else 0.);den=va+vb+(ma-mb)**2;con.append(2*cov/den if den else 0.)
  for key,value in [('per_channel_pearson',pear),('per_channel_concordance',con)]:maxerr=max(maxerr,float(abs(np.array(value)-row['metrics'][key]).max(initial=0)))
  checked+=1
assert maxerr<1e-9,maxerr
p=read(OUT/'raw_partner_protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for n,h in p['sources'].items():assert sha(OUT/n)==h,n
with (OUT/'raw_partner_audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,checked_pairs=checked,max_scalar_moment_error=maxerr,shared='Labels/maps/coordinates same as main registered GT oracle; independent scalar moments.'),f,indent=2)
