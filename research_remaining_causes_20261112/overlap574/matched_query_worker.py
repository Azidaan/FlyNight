import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.z13_readout import HexGeometry
from tracking.channel_balanced_correspondence import _center_pair
from tracking.channel_agreement_correspondence import agreement_channels
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 p=read(OUT/'matched_query_protocol.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
 for n,h in p['sources'].items():assert sha(OUT/n)==h,n
 return p
assert (OUT/'GO_matched.json').exists(),'Await parent registration'
p=verify();old=read(OUT/'result.json');heads=old['heads'];stage=read(ROOT/'research_stage_localization_20261112/protocol.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;rows=[];witness=[];samples={};meta=[];maxerr=0.
def evaluate(a,b,geometry,arm,head):
 if len(a)<3:return None
 if head=='agreement':q=agreement_channels(a,b,mean_penalty=True)
 else:
  x,y,an,bn,active,activeb=_center_pair(a,b)
  if not active.any() or not activeb.any():return None
  q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
 if q is None:return None
 h=heads[arm][head];features=np.r_[(np.array(geometry)-h['geometry']['mean'])/h['geometry']['scale'],(q-h['signal_mean'])/h['signal_scale']];t=np.array(h['theta']);return float(features@t[:-1]+t[-1])
def inside(x,b):return ((x>=b[:2])&(x<=b[2:])).all(1)
for ci,c in enumerate(stage['cases']):
 r=c['case'];o=next(o for o in old['meta'] if o['old_id']==r['old_id']);boxes=[np.array([o['owners'][ei]['box2d'][k] for k in ['x1','y1','x2','y2']]) for ei in [0,1]];refbb,curbb=boxes;rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);plan=relative_template_plan(reader,r['reference_box'],rc,r['current_box'],cc,r['offsets']);cells=plan['cells'];world=reader.points[cells]/reader.eye_size*(rc[2:]-rc[:2])+rc[:2];zero=np.flatnonzero((np.array(r['offsets'])==0).all(1));assert len(zero)==1
 for arm in p['representations']:
  maps=[]
  for ep in c['endpoints']:
   m=np.load(ROOT/ep['paths']['C34' if arm=='C34' else 'R1']).astype(float)
   if arm=='R134_spatial':m=np.c_[m[:,:17],m[nb[:,1:],:17].mean(1)]
   maps.append(m)
  queries={'original':plan['query'][:,r['old_variants'][arm]['winner']],'detector_zero':plan['query'][:,int(zero[0])]};affine=curbb[:2]+(world-refbb[:2])/(refbb[2:]-refbb[:2])*(curbb[2:]-curbb[:2]);center=world+(curbb[:2]+curbb[2:]-refbb[:2]-refbb[2:])/2
  for mode,w in [('gt_affine_zero',affine),('gt_center_zero',center)]:queries[mode]=(w-cc[:2])/(cc[2:]-cc[:2])*reader.eye_size
  common=inside(world,refbb);dual=common.copy()
  for q in queries.values():
   common &= reader.mesh.find_simplex(q)>=0;w=q/reader.eye_size*(cc[2:]-cc[:2])+cc[:2];dual &= inside(w,curbb)&(reader.mesh.find_simplex(q)>=0)
  key=f'{ci}_{arm}';samples[key+'_a_full']=maps[0][cells];samples[key+'_map_current']=maps[1];samples[key+'_reference_world']=world;samples[key+'_common_referenceGT']=common;samples[key+'_common_dualGT']=dual;samples[key+'_cells']=cells
  for head in p['heads']:
   value=evaluate(maps[0][cells],sample_linear(reader.mesh,maps[1],queries['original']),r['geometry'],arm,head);expected=next(x['score'] for x in old['records'] if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==head and x['mode']=='full');assert abs(value-expected)<1e-9;witness.append(dict(old_id=r['old_id'],representation=arm,head=head,value=value,expected=expected,error=abs(value-expected)))
  for mode,q in queries.items():
   samples[key+'_'+mode+'_query']=q
   for support,mask in [('common_referenceGT',common),('common_dualGT',dual)]:
    a=maps[0][cells][mask].copy();b=sample_linear(reader.mesh,maps[1],q[mask]);independent=LinearNDInterpolator(reader.mesh,maps[1])(q[mask]);maxerr=max(maxerr,float(abs(b-independent).max(initial=0)));samples[key+'_'+support+'_'+mode+'_b']=b
    for head in p['heads']:rows.append(dict(old_id=r['old_id'],case=ci,label=r['label'],representation=arm,head=head,support=support,mode=mode,count=len(a),score=evaluate(a,b,r['geometry'],arm,head)))
  meta.append(dict(case=ci,old_id=r['old_id'],representation=arm,geometry=r['geometry'],reference_gt=refbb.tolist(),current_gt=curbb.tolist(),current_crop=cc.tolist(),original_winner=r['old_variants'][arm]['winner'],original_offset=r['offsets'][r['old_variants'][arm]['winner']]))
 print('MATCHED',r['old_id'],flush=True)
assert maxerr<1e-9;verify()
with (OUT/'matched_query_samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
with (OUT/'matched_query_result.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,records=rows,witness=witness,meta=meta,heads=heads,max_independent_interpolation_error=maxerr,samples_sha256=sha(OUT/'matched_query_samples.npz'),core_calls=0,new_fits=0,tracker_calls=0),f,ensure_ascii=False,indent=2)
