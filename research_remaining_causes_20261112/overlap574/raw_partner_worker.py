import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math,sys
from pathlib import Path
import numpy as np,cv2
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.hex_linear_interpolation import sample_linear
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (OUT/'GO_raw.json').exists(),'Await raw partner registration'
p=read(OUT/'raw_partner_protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for n,h in p['sources'].items():assert sha(OUT/n)==h,n
stage=read(ROOT/'research_stage_localization_20261112/protocol.json');case=next(c for c in stage['cases'] if c['case']['old_id']==574);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);frames=[np.array(Image.open(ROOT/f'research_failure_localization_20261112/frames/574_{f}_full.png').convert('RGB')) for f in [931,937]];grays=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in frames];inputs=[np.load(ROOT/case['views'][ep['length']-1]).astype(float).reshape(721,1) for ep in case['endpoints']];maxerr=0.;rows=[];arrays={}
def bilinear(image,xy):
 global maxerr
 image=image.astype(float);image=image[:,:,None] if image.ndim==2 else image;x=xy[:,0];y=xy[:,1];ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);u=(x-ix)[:,None];v=(y-iy)[:,None];value=(1-u)*(1-v)*image[iy,ix]+u*(1-v)*image[iy,ix+1]+(1-u)*v*image[iy+1,ix]+u*v*image[iy+1,ix+1];other=RegularGridInterpolator((np.arange(image.shape[0]),np.arange(image.shape[1])),image)(np.c_[y,x]);maxerr=max(maxerr,float(abs(value-other).max(initial=0)));return value
def metrics(a,b):
 if len(a)<3:return None
 ma=a.mean(0);mb=b.mean(0);x=a-ma;y=b-mb;va=(x*x).mean(0);vb=(y*y).mean(0);cov=(x*y).mean(0);den=va+vb+(ma-mb)**2;pear=np.divide(cov,np.sqrt(va*vb),out=np.zeros(a.shape[1]),where=va*vb>0);con=np.divide(2*cov,den,out=np.zeros(a.shape[1]),where=den>0);return dict(pearson=float(pear.mean()),concordance=float(con.mean()),per_channel_pearson=pear.tolist(),per_channel_concordance=con.tolist())
main=read(OUT/'result.json')
with np.load(OUT/'samples.npz') as saved:
 for row in main['oracle_records']:
  if row['head']!='native':continue
  k=row['sample_key'];xy=[saved[k+'_reference_world'],saved[k+'_current_world']];stages={row['representation']:(saved[k+'_a'].copy(),saved[k+'_b'].copy())}
  if row['representation']=='C34':
   stages.update(raw_RGB=tuple(bilinear(im,x) for im,x in zip(frames,xy)),raw_gray=tuple(bilinear(im,x) for im,x in zip(grays,xy)))
   query=[(x-np.array(crop[:2]))/(np.array(crop[2:])-crop[:2])*reader.eye_size for x,crop in zip(xy,[case['case']['reference_crop'],case['case']['current_crop']])];retina=tuple(sample_linear(reader.mesh,m,q) for m,q in zip(inputs,query));stages['prepared_retina']=retina
   for m,q,a in zip(inputs,query,retina):maxerr=max(maxerr,float(abs(a-LinearNDInterpolator(reader.mesh,m)(q)).max(initial=0)))
  for name,(a,b) in stages.items():
   key=k+'_'+name;arrays[key+'_a']=a;arrays[key+'_b']=b;rows.append(dict(mapping=row['mapping'],support=row['support'],partner_id=row['partner_id'],is_own=row['is_own'],stage=name,count=len(a),sample_key=key,metrics=metrics(a,b)))
assert maxerr<1e-9,maxerr
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with (OUT/'raw_partner_samples.npz').open('xb') as f:np.savez_compressed(f,**arrays)
with (OUT/'raw_partner_result.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=rows,max_independent_interpolation_error=maxerr,samples_sha256=sha(OUT/'raw_partner_samples.npz'),core_calls=0,new_fits=0,tracker_calls=0),f,indent=2)
