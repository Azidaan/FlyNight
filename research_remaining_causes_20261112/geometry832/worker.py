import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import Delaunay
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def cast(x):
 if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cast(v) for v in x]
 return x.tolist() if hasattr(x,'tolist') else x
def save(name,x):
 with (OUT/name).open('x',encoding='utf8') as f:json.dump(cast(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
 for n,h in p['inputs'].items():assert sha(n)==h,n
def gt_box(video,frame,box):
 labels=read(f'data/bdd100k/labels/box_track_20/val/{video}.json');clock=read(f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json');anchor=clock['anchors'][str(frame)];by={int(x.get('frameIndex',i)):x for i,x in enumerate(labels)};row=by[anchor['frameIndex']];assert row['name']==anchor['name'];hits=[];a=np.array(box)
 for item in row.get('labels') or []:
  if item['category']!='car' or (item.get('attributes') or {}).get('crowd',False):continue
  b=np.array([item['box2d'][k] for k in ['x1','y1','x2','y2']]);I=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();U=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-I;hits.append((float(I/U),str(item['id']),b))
 hit=max(hits,key=lambda x:x[0]);assert hit[0]>=.5,(video,frame,hit[:2]);return dict(identity=hit[1],iou=hit[0],box=hit[2],raw_frame=frame,annotation_frameIndex=anchor['frameIndex'],name=anchor['name'])
def channels(a,b,head):
 if head=='agreement':return agreement_channels(a,b,mean_penalty=True)
 x,y,an,bn,active,_=_center_pair(a,b);return np.divide((x*y).sum(0),an*bn,out=np.zeros(a.shape[1]),where=an*bn>0)*a.shape[1]/active.sum()
def score(q,geom,h):
 features=np.r_[(np.array(geom)-h['geometry']['mean'])/h['geometry']['scale'],(q-h['signal_mean'])/h['signal_scale']];t=np.array(h['theta']);return float(features@t[:-1]+t[-1])
p=read(str((OUT/'proposed_protocol.json').relative_to(ROOT)));assert (OUT/'GO.json').exists(),'Parent registration and GO required';go=json.loads((OUT/'GO.json').read_text(encoding='utf8'));assert go['approved'] is True;verify(p)
deadline=datetime.fromisoformat(p['runtime']['deadline']);started=time.monotonic();assert datetime.now(ZoneInfo('Europe/Moscow'))<deadline
g=read('runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);stage=read('research_stage_localization_20261112/protocol.json');heads=read('research_candidate_support_probe_20261112/scores.json')['heads'];old=read('research_input_precision_probe_20261112/scores.json')['records'];rows=[];meta=[];arrays={};baseerr=0.
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as maps,np.load(ROOT/'research_input_precision_probe_20261112/expanded_inputs.npz') as inputs:
 for ci,c in enumerate(stage['cases']):
  assert time.monotonic()-started<180 and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
  r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);x=reader.points[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];ref=gt_box(r['video'],c['sequence'][c['endpoints'][0]['length']-1][0],r['reference_box']);cur=gt_box(r['video'],r['frame'],r['current_box']);a=ref['box'];b=cur['box'];center=x+(b[:2]+b[2:]-a[:2]-a[2:])/2;affine=b[:2]+(x-a[:2])*(b[2:]-b[:2])/(a[2:]-a[:2]);rb=np.array(r['reference_box']);cb=np.array(r['current_box']);det=cb[:2]+(x-rb[:2])*(cb[2:]-cb[:2])/(rb[2:]-rb[:2]);queries={};reach={}
  for target,name in [(center,'gt_center'),(affine,'gt_affine')]:
   req=target-det;off=np.array(r['offsets']);inside=((req>=off.min(0))&(req<=off.max(0))).all(1);dist=np.linalg.norm(req[:,None,:]-off,axis=2).min(1);reach[name]=dict(required_offset_min=req.min(0),required_offset_max=req.max(0),allowed_continuous_min=off.min(0),allowed_continuous_max=off.max(0),cells_reachable_in_continuous_rectangle=int(inside.sum()),cells=len(cells),min_grid_distance=float(dist.min()),max_grid_distance=float(dist.max()),required_offset=req)
  for queryarm in ['C34','R134_temporal']:
   off=np.array(r['offsets'][r['old_variants'][queryarm]['winner']]);worlds=dict(detector_affine_saved_offset=det+off,gt_center_saved_offset=center+off,gt_affine_saved_offset=affine+off,gt_center_zero_offset=center,gt_affine_zero_offset=affine)
   for name,world in worlds.items():queries[(queryarm,name)]=(world-cc[:2])*reader.eye_size/(cc[2:]-cc[:2])
   assert np.max(abs(queries[(queryarm,'detector_affine_saved_offset')]-plan['query'][:,r['old_variants'][queryarm]['winner']]))<1e-9
  common=np.ones(len(cells),bool)
  for q in queries.values():common&=reader.mesh.find_simplex(q)>=0
  assert common.sum()>=3
  meta.append(dict(old_id=r['old_id'],label=r['label'],geometry=r['geometry'],reference=ref,current=cur,full_count=len(cells),common_count=int(common.sum()),reach=reach))
  arrays[f'{ci}_cells']=cells;arrays[f'{ci}_common']=common;arrays[f'{ci}_world_source']=x
  for arm in ['C34','R134_temporal','retina32']:
   arr=[]
   for ei,ep in enumerate(c['endpoints']):
    arr.append(inputs[f'precision32_{ci}'][ep['length']-1,:,None].astype(float) if arm=='retina32' else maps[f'index_{ci}_{ei}_'+('C34' if arm=='C34' else 'R1')].astype(float))
    arrays[f'{ci}_{arm}_map{ei}']=arr[-1]
   for (queryarm,variant),q in queries.items():
    for support,M in [('common',common),('full_identity_witness',np.ones(len(cells),bool))]:
     if support=='full_identity_witness' and variant!='detector_affine_saved_offset':continue
     A=arr[0][cells][M];B=sample_linear(reader.mesh,arr[1],q[M]);key=f'{ci}_{arm}_{queryarm}_{variant}_{support}';arrays[key+'_a']=A;arrays[key+'_b']=B;arrays[key+'_q']=q[M]
     for head in ['native','agreement']:
      ch=channels(A,B,head);s=None if arm=='retina32' else score(ch,r['geometry'],heads[arm][head]);rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,query=queryarm,variant=variant,support=support,count=len(A),head=head,channels=ch,mean_similarity=float(ch.mean()),score=s,accepted=None if s is None else bool(s>0),patch_key=key))
      if support=='full_identity_witness' and queryarm==arm and arm!='retina32':
       baseline=next(z for z in old if z['old_id']==r['old_id'] and z['mode']=='index' and z['representation']==arm and z['head']==head);err=max(abs(s-baseline['score']),float(abs(ch-baseline['channels']).max()));baseerr=max(baseerr,err);assert err<1e-9,(r['old_id'],arm,head,err)
  (OUT/'status.json').write_text(json.dumps(dict(old_id=r['old_id'],stage='measured',updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat())),encoding='utf8')
verify(p)
with (OUT/'paired_samples.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('result.json',dict(passed=True,records=rows,geometry=meta,max_baseline_error=baseerr,new_core_calls=0,new_fits=0,new_tracker_calls=0,paired_samples_sha256=hashlib.sha256((OUT/'paired_samples.npz').read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started))
print('COMPLETE',len(rows),'baseline',baseerr)
