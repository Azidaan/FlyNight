"""Independent scalar GT-clock, transform, and continuous-domain check.

This does not import the worker or production coordinate helpers. GT boxes are
diagnostic anchors; their corresponding normalized fractions are not physical
landmarks, especially for foreign objects and changing visibility.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def best_annotation(video,rawframe,box):
 labels=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json');clock=read(ROOT/f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json');anchor=clock['anchors'][str(rawframe)];matches=[r for i,r in enumerate(labels) if int(r.get('frameIndex',i))==int(anchor['frameIndex'])];assert len(matches)==1;frame=matches[0];assert frame['name']==anchor['name'];found=[]
 for obj in frame.get('labels') or []:
  if obj['category']!='car' or (obj.get('attributes') or {}).get('crowd',False):continue
  b=[float(obj['box2d'][k]) for k in ['x1','y1','x2','y2']];I=math.prod(max(0.,min(box[j+2],b[j+2])-max(box[j],b[j])) for j in [0,1]);U=math.prod(box[j+2]-box[j] for j in [0,1])+math.prod(b[j+2]-b[j] for j in [0,1])-I;found.append((I/U,str(obj['id']),b))
 hit=max(found,key=lambda z:z[0]);assert hit[0]>=.5;return hit
p=read(OUT/'proposed_protocol.json');res=read(OUT/'result.json');stage=read(ROOT/'research_stage_localization_20261112/protocol.json');points=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');metadata={r['old_id']:r for r in res['geometry']};maxerr=0.;evidence=[]
with np.load(OUT/'paired_samples.npz') as samples:
 for ci,c in enumerate(stage['cases']):
  r=c['case'];m=metadata[r['old_id']];refraw=c['sequence'][c['endpoints'][0]['length']-1][0];oldgt=best_annotation(r['video'],refraw,r['reference_box']);newgt=best_annotation(r['video'],r['frame'],r['current_box']);assert oldgt[1]==m['reference']['identity'] and newgt[1]==m['current']['identity'];assert oldgt[2]==m['reference']['box'] and newgt[2]==m['current']['box'];reach={};coords={}
  for qa in ['C34','R134_temporal']:
   offset=r['offsets'][r['old_variants'][qa]['winner']]
   for variant in p['interventions']:
    dst=[];required=[];cells=samples[f'{ci}_cells']
    for cell in cells:
     xy=[float(points['points'][int(cell)][j])*(r['reference_crop'][j+2]-r['reference_crop'][j])/points['eye_size']+r['reference_crop'][j] for j in [0,1]];base=[r['current_box'][j]+(xy[j]-r['reference_box'][j])*(r['current_box'][j+2]-r['current_box'][j])/(r['reference_box'][j+2]-r['reference_box'][j]) for j in [0,1]]
     if variant.startswith('detector_'):world=base
     elif variant.startswith('gt_center_'):world=[xy[j]+(newgt[2][j]+newgt[2][j+2]-oldgt[2][j]-oldgt[2][j+2])/2 for j in [0,1]]
     else:world=[newgt[2][j]+(xy[j]-oldgt[2][j])*(newgt[2][j+2]-newgt[2][j])/(oldgt[2][j+2]-oldgt[2][j]) for j in [0,1]]
     if variant!='detector_affine_saved_offset':required.append([world[j]-base[j] for j in [0,1]])
     if variant.endswith('saved_offset'):world=[world[j]+offset[j] for j in [0,1]]
     dst.append([(world[j]-r['current_crop'][j])*points['eye_size']/(r['current_crop'][j+2]-r['current_crop'][j]) for j in [0,1]])
    coords[(qa,variant)]=np.array(dst)
    if qa=='C34' and variant.endswith('zero_offset'):
     name='gt_center' if 'center' in variant else 'gt_affine';lo=[min(s[j] for s in r['offsets']) for j in [0,1]];hi=[max(s[j] for s in r['offsets']) for j in [0,1]];inside=[all(lo[j]<=d[j]<=hi[j] for j in [0,1]) for d in required];dist_rectangle=[math.sqrt(sum(max(lo[j]-d[j],0.,d[j]-hi[j])**2 for j in [0,1])) for d in required];distgrid=[min(math.dist(d,s) for s in r['offsets']) for d in required];reach[name]=dict(cells=len(cells),continuous_rectangle_reachable=sum(inside),unreachable_by_any_allowed_offset=len(cells)-sum(inside),required_min=[min(d[j] for d in required) for j in [0,1]],required_max=[max(d[j] for d in required) for j in [0,1]],allowed_min=lo,allowed_max=hi,min_distance_to_continuous_rectangle=min(dist_rectangle),max_distance_to_continuous_rectangle=max(dist_rectangle),min_distance_to_discrete_grid=min(distgrid),max_distance_to_discrete_grid=max(distgrid),proof='A point outside the bounding rectangle is outside every allowed discrete shift and every continuous domain contained in that rectangle. Inside rectangle does not prove reachable in the hexagonal search domain.')
     assert reach[name]['continuous_rectangle_reachable']==m['reach'][name]['cells_reachable_in_continuous_rectangle']
  for row in res['records']:
   if row['old_id']!=r['old_id']:continue
   M=samples[f'{ci}_common'] if row['support']=='common' else np.ones(len(samples[f'{ci}_cells']),bool);expected=coords[(row['query'],row['variant'])][M];actual=samples[row['patch_key']+'_q'];maxerr=max(maxerr,float(abs(expected-actual).max()))
  evidence.append(dict(old_id=r['old_id'],reference_raw_frame=refraw,current_raw_frame=r['frame'],reference_identity=oldgt[1],current_identity=newgt[1],same_annotation_identity=oldgt[1]==newgt[1],reach=reach))
assert maxerr<1e-9,maxerr
out=dict(passed=True,max_coordinate_error=maxerr,evidence=evidence,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),independence='Scalar formulas; rawframe-to-annotation clocks independently read; no worker or production transform imports.',limits='Proves relation of specified GT-box coordinate maps to the entire bounded search domain, not physical-part correspondence, neural failure causation, operational recovery or HOTA.')
with (OUT/'coordinate_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print('COORDINATE AUDIT',maxerr)
