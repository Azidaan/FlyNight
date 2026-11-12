import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify():
 p=read(OUT/'proposed_protocol.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
 for n,h in p['source_hashes'].items():assert sha(OUT/n)==h,n
 return p
assert (OUT/'GO.json').exists(),'Await parent experiment registration and go'
p=verify();start=time.monotonic()
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard=guard_entrypoint(timeout=600)
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.z13_readout import HexGeometry
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors
stage=read(ROOT/'research_stage_localization_20261112/protocol.json');previous=read(ROOT/'research_candidate_support_probe_20261112/scores.json');heads=previous['heads'];events=read(ROOT/'research_failure_localization_20261112/stage1.json')['records'];rows=[];controls=[];samples={};metadata=[];baseline=[];oracle=[]
def evaluate(a,b,geometry,arm,head):
 if len(a)<3:return None
 if head=='agreement':q=agreement_channels(a,b,mean_penalty=True)
 else:
  x,y,an,bn,active,active_b=_center_pair(a,b)
  if not active.any() or not active_b.any():return None
  q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
 if q is None:return None
 h=heads[arm][head];geometry=(np.array(geometry)-h['geometry']['mean'])/h['geometry']['scale'];features=np.r_[geometry,(q-h['signal_mean'])/h['signal_scale']];t=np.array(h['theta'])
 return dict(score=float(features@t[:-1]+t[-1]),channels=q.tolist())
def ann(v,frame):
 labels=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');clock=read(ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json')['anchors'][str(frame)];row=next(r for i,r in enumerate(labels) if int(r.get('frameIndex',i))==clock['frameIndex']);assert row['name']==clock['name'];return row.get('labels') or [],clock
def box(o):return np.array([o['box2d'][k] for k in ['x1','y1','x2','y2']])
def inside(x,bb):return ((x>=bb[:2])&(x<=bb[2:])).all(1)
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as maps:
 for ci,c in enumerate(stage['cases']):
  r=c['case'];v=r['video'];fr=[c['sequence'][ep['length']-1][0] for ep in c['endpoints']];e=next(e for e in events if e['old_id']==r['old_id'] and e['video']==v and e['frame']==r['frame']);ids=[e['reference_hits'][0]['identity'],e['current_hits'][0]['identity']];objects=[];clocks=[]
  for frame in fr:
   objects_,clock=ann(v,frame);objects.append(objects_);clocks.append(clock)
  owners=[next(o for o in objects[i] if str(o['id'])==ids[i]) for i in [0,1]];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];rc=np.array(r['reference_crop']);refworld=reader.points[cells]/reader.eye_size*(rc[2:]-rc[:2])+rc[:2]
  metadata.append(dict(case=ci,old_id=r['old_id'],video=v,frames=fr,ids=ids,clocks=clocks,owners=owners,objects=objects,geometry=r['geometry']))
  for ai,arm in enumerate(p['representations']):
   assert time.monotonic()-start<570
   arrays=[]
   for ei in [0,1]:
    m=maps[f'index_{ci}_{ei}_C34'].astype(float) if arm=='C34' else maps[f'index_{ci}_{ei}_R1'].astype(float)
    if arm=='R134_spatial':m=np.c_[m[:,:17],m[nb[:,1:],:17].mean(1)]
    arrays.append(m)
   query=plan['query'][:,r['old_variants'][arm]['winner']];cc=np.array(r['current_crop']);world=query/reader.eye_size*(cc[2:]-cc[:2])+cc[:2];a=arrays[0][cells];b=sample_linear(reader.mesh,arrays[1],query);key=f'{ci}_{arm}'
   dual=inside(refworld,box(owners[0]))&inside(world,box(owners[1]));exclusive=dual.copy();ambiguous=[]
   for ei,xy in enumerate([refworld,world]):
    union=np.zeros(len(a),bool)
    for o in objects[ei]:
     if str(o['id'])!=ids[ei] and 'box2d' in o:union|=inside(xy,box(o))
    ambiguous.append(union);exclusive&=~union
   for n,arr in dict(a=a,b=b,cells=cells,reference_world=refworld,current_world=world,dual_gt=dual,dual_exclusive=exclusive,ambiguous_reference=ambiguous[0],ambiguous_current=ambiguous[1]).items():samples[key+'_'+n]=arr
   masks={'full':np.ones(len(a),bool),'identity':np.ones(len(a),bool),'dual_gt':dual,'dual_exclusive':exclusive}
   for mode,mask in masks.items():
    for head in p['heads']:
     z=evaluate(a[mask].copy(),b[mask].copy(),r['geometry'],arm,head);row=dict(old_id=r['old_id'],case=ci,representation=arm,head=head,mode=mode,count=int(mask.sum()),label=r['label'],score=None if z is None else z['score'],channels=None if z is None else z['channels']);rows.append(row)
     if mode=='full':
      expected=next(s['score'] for s in previous['records'] if s['old_id']==r['old_id'] and s['representation']==arm and s['head']==head and s['mode']=='full');assert z is not None and abs(z['score']-expected)<1e-9;baseline.append(dict(old_id=r['old_id'],representation=arm,head=head,error=abs(z['score']-expected)))
     if mode=='identity':assert row['score']==next(x['score'] for x in rows if x['old_id']==r['old_id'] and x['representation']==arm and x['head']==head and x['mode']=='full')
   rng=np.random.default_rng(57420261112+ci*100+ai)
   for mode,parent in [('dual_gt',np.arange(len(a))),('dual_exclusive',np.flatnonzero(dual))]:
    k=int(masks[mode].sum());N=len(parent)
    if N==0:continue
    indices=np.array([parent[(np.arange(k)+j)%N] for j in range(N)]+[np.sort(rng.choice(parent,k,replace=False)) for _ in range(199)],dtype=int);samples[key+'_'+mode+'_control_indices']=indices
    for head in p['heads']:
     values=[evaluate(a[ix],b[ix],r['geometry'],arm,head) for ix in indices];controls.append(dict(old_id=r['old_id'],case=ci,representation=arm,head=head,mode=mode,count=k,parent_count=N,cyclic_count=N,random_count=199,scores=[None if z is None else z['score'] for z in values]))
   if r['old_id']==574:
    # Same archived current eye contains both candidate IDs. Geometry covariates
    # remain those of original false pair to isolate paired visual evidence.
    oldbb=box(owners[0]);partners=[next(o for o in objects[1] if str(o['id'])==identity) for identity in [ids[0],ids[1]]]
    for mapping in ['gt_center','gt_affine']:
     queries=[];worlds=[];common=inside(refworld,oldbb);exclusive_common=common.copy()
     for partner in partners:
      bb=box(partner)
      w=refworld+(bb[:2]+bb[2:]-oldbb[:2]-oldbb[2:])/2 if mapping=='gt_center' else bb[:2]+(refworld-oldbb[:2])/(oldbb[2:]-oldbb[:2])*(bb[2:]-bb[:2])
      q=(w-cc[:2])/(cc[2:]-cc[:2])*reader.eye_size;queries.append(q);worlds.append(w);common &= inside(w,bb)&(reader.mesh.find_simplex(q)>=0)
     exclusive_common=common.copy()
     for ei,xy0,owner_id in [(0,refworld,ids[0]),(1,worlds[0],ids[0]),(1,worlds[1],ids[1])]:
      for other in objects[ei]:
       if str(other['id'])!=owner_id and 'box2d' in other:exclusive_common &= ~inside(xy0,box(other))
     for support,mask in [('common_gt',common),('common_exclusive',exclusive_common)]:
      for partner,q,w in zip(partners,queries,worlds):
       okey=key+'_'+mapping+'_'+support+'_'+str(partner['id']);aa=a[mask].copy();bb=sample_linear(reader.mesh,arrays[1],q[mask]);samples[okey+'_a']=aa;samples[okey+'_b']=bb;samples[okey+'_reference_world']=refworld[mask];samples[okey+'_current_world']=w[mask];samples[okey+'_indices']=np.flatnonzero(mask)
       for head in p['heads']:
        z=evaluate(aa,bb,r['geometry'],arm,head);oracle.append(dict(old_id=574,case=ci,representation=arm,head=head,mapping=mapping,support=support,partner_id=str(partner['id']),is_own=str(partner['id'])==ids[0],sample_key=okey,count=len(aa),score=None if z is None else z['score'],channels=None if z is None else z['channels']))
   print('CASE',r['old_id'],arm,flush=True)
verify()
with (OUT/'samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
save('result.json',dict(passed=True,records=rows,oracle_records=oracle,controls=controls,meta=metadata,heads=heads,baseline=baseline,samples_sha256=sha(OUT/'samples.npz'),new_fits=0,core_calls=0,tracker_calls=0,elapsed_seconds=time.monotonic()-start))
print('COMPLETE',len(rows),flush=True)
