from common import *
import numpy as np
p=verify();r=read(OUT/'result.json');assert r['passed'];checks=[];max_cost=0.
assert sha(OUT/'probe.py')==p['inputs'][(OUT/'probe.py').relative_to(ROOT).as_posix()]
for t in r['transforms']:
 x=np.array(t['points']);y=np.array(t['destinations']);H=np.array(t['H']);train=np.array(t['train']);errors=[]
 for src,dst in zip(x,y):
  px=H[0,0]*src[0]+H[0,1]*src[1]+H[0,2];py=H[1,0]*src[0]+H[1,1]*src[1]+H[1,2];errors.append(((px-dst[0])**2+(py-dst[1])**2)**.5)
 assert np.max(abs(np.array(errors)-np.array(t['errors'])))<1e-10
 for pts in [x,y]:assert not np.any((pts[:,0]>=900)&(pts[:,0]<=1100)&(pts[:,1]>=475)&(pts[:,1]<=620))
 held=np.array(errors)[~train];assert abs(np.quantile(held,.9)-t['heldout_p90'])<1e-10
 checks.append(dict(frames=t['frames'],heldout_p90=t['heldout_p90'],ready=t['ready']))
native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json')['videos'][p['video']]
with np.load(ROOT/native['detections']) as z:raw={k:z[k] for k in z.files}
# Scalar independent geometry, matching the float32 sequence at each operation.
bs=[]
for rid in r['raw_ids']:
 b=raw['xyxy'][rid];cx=np.float32((b[0]+b[2])/2);cy=np.float32((b[1]+b[3])/2);w=np.float32(b[2]-b[0]);h=np.float32(b[3]-b[1]);x=np.float32(float(cx)-float(w)/2);y=np.float32(float(cy)-float(h)/2);bs.append([x,y,np.float32(x+w),np.float32(y+h)])
for a in r['arms']:
 for i,box in enumerate(a['boxes']):
  x1,y1,x2,y2=np.array(box,np.float32)
  for j,b in enumerate(bs):
   X1,Y1,X2,Y2=b;inter=np.float32(max(0,min(x2,X2)-max(x1,X1)))*np.float32(max(0,min(y2,Y2)-max(y1,Y1)))
   area=np.float32(x2-x1)*np.float32(y2-y1);B=np.float32(X2-X1)*np.float32(Y2-Y1)
   # bbox_ioa adds1e-7 after union, all in float32.
   ov=np.float32(inter/np.float32(np.float32(area+B-inter)+np.float32(1e-7)))
   dist=np.float32(1-ov);cost=np.float32(1-np.float32(np.float32(1-dist)*raw['conf'][r['raw_ids'][j]]))
   err=abs(float(cost)-a['matrix'][i][j]);max_cost=max(max_cost,err);assert err==0,(a['name'],i,j,err)
 full=next(x for x in r['arms'] if x['name']=='full_all');res=next(x for x in r['arms'] if x['name']=='residual_all')
 row232=r['row_ids'].index(232);row474=r['row_ids'].index(474);col=r['raw_ids'].index(54471)
case=dict(full232=full['matrix'][row232][col],full474=full['matrix'][row474][col],residual232=res['matrix'][row232][col],residual474=res['matrix'][row474][col],full_to_residual_corner_max_pixels=float(np.max(np.abs(np.array(full['boxes'])[row232]-np.array(res['boxes'])[row232]))))
save('audit.json',dict(passed=True,input_unchanged=True,source_verified=True,scalar_costs_exact=True,cost_entries=len(r['arms'])*len(r['row_ids'])*len(r['raw_ids']),max_cost_error=max_cost,background_metrics_verified=checks,readiness_passed=r['readiness_passed'],competition=case,viewed=['background_538.jpg','background_539.jpg'],sensitivity=r['sensitivity'],visual_observation='All fitted and held-out points lie in the fixed facade region; target excluded. Same depth as target is not established.',scope='Arithmetic and measured image displacements verified. Camera-only sufficiency and true identity not established.'))
print('AUDIT',case)
