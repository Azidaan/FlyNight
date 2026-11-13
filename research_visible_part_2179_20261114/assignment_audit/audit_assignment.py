import os,json,hashlib,gzip,math
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'.cache/tmp')
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
cv2.setNumThreads(1)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
prev=ROOT/'research_assignment_weights_2179_20261112';ep=read(prev/'evaluation_plan.json');clock=read(ROOT/ep['clock'])['anchors'];labels=read(ROOT/ep['labels']);by={int(x.get('frameIndex',i)):x for i,x in enumerate(labels)}
runtime=read(prev/'runtime.json'); source=ROOT/ep['source_archive']
with gzip.open(source,'rt',encoding='utf8') as f:events=json.load(f)['events']
with np.load(ROOT/ep['detector_file'],allow_pickle=False) as f:raw={k:f[k].copy() for k in ['xyxy','conf','cls','frame_idx']}
def iou(a,b):
 a=list(map(float,a));b=list(map(float,b));x=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x;return x/u if u>0 else 0
def gt(frame):
 if str(frame) not in clock:return None
 x=clock[str(frame)];r=by[x['frameIndex']];assert r['name']==x['name'];return r
def hits(frame,box):
 row=gt(frame)
 if row is None:return None
 return sorted([dict(identity=str(o['id']),iou=iou(box,[o['box2d'][k] for k in ['x1','y1','x2','y2']]),category=o['category']) for o in row.get('labels',[]) if o.get('box2d') and not o.get('attributes',{}).get('crowd',False)],key=lambda x:-x['iou'])[:6]
def solve(cost):
 n,m=cost.shape;a=np.zeros((n+m,n+m));a[:n,:m]=cost;a[:n,m:]=.4;a[n:,:m]=.4;ii,jj=linear_sum_assignment(a);return [(int(i),int(j)) for i,j in zip(ii,jj) if i<n and j<m]
records=[]
for e in runtime['events']:
 ids=[x['tid'] for x in e['rows']];rawids=[x['raw_id'] for x in e['candidates']];a=np.asarray(e['corrected'],float);base=np.asarray(e['base'],float);pairs=solve(a);assert sorted(pairs)==sorted(map(tuple,e['actual_first']))
 if e['frame']==1159:continue
 rid=109699 if e['frame']==1160 else 110260;j=rawids.index(rid);gap=base[ids.index(2179),j]-base[ids.index(2445),j];delta=(a-base)[[ids.index(2179),ids.index(2445)],j]
 checks={}
 for remove in [False,True]:
  for classmask in [False,True]:
   keep=[i for i,t in enumerate(ids) if not(remove and t==2445)];c=a[keep].copy()
   if classmask:
    for ri,i in enumerate(keep):
     for k,col in enumerate(e['candidates']):
      if e['rows'][i]['cls']!=col['cls']:c[ri,k]=1000
   assigned=[(ids[keep[i]],rawids[k]) for i,k in solve(c)];checks[f'remove2445={remove},classmask={classmask}']=dict(owner=[t for t,r in assigned if r==rid],target=[r for t,r in assigned if t==2179])
 records.append(dict(frame=e['frame'],raw_id=rid,current_hits=hits(e['frame'],raw['xyxy'][rid]),cost_column=sorted([dict(tid=t,cls=e['rows'][i]['cls'],cost=a[i,j],base=base[i,j]) for i,t in enumerate(ids)],key=lambda x:x['cost'])[:8],candidate_class=e['candidates'][j]['cls'],archived_target_deltas=delta.tolist(),geometry_gap=float(gap),minimal_pair_gap_at_alpha025=float(.75*gap-.25+delta[0]-delta[1]),alpha_for_pair_flip_with_ideal_appearance=float(gap/(1+gap)),counterfactuals=checks))
anchors=[]
for frame in [1153,1159,1165]:
 event=next(x for x in events if x['frame']==frame);ids={t:event['assigned'].get(str(t)) for t in [2179,2445]};row=gt(frame)
 anchors.append(dict(frame=frame,anchor=clock[str(frame)],objects=[o for o in row['labels'] if str(o['id']) in ['00122657','00122746']],assignments=[dict(tid=t,raw_id=r,hits=hits(frame,raw['xyxy'][r]) if r is not None else None) for t,r in ids.items()]))
video=ROOT/'data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov';cap=cv2.VideoCapture(str(video));frames={}
for frame in range(1172):
 ok,img=cap.read();assert ok
 if 1148<=frame<=1171:frames[frame]=img.copy()
cap.release();image_alignment=[]
for frame in [1153,1159,1165]:
 row=gt(frame);jpgpath=ROOT/'data/bdd100k/images/track/val/b1ca2e5d-84cf9134'/row['name'];img=cv2.imread(str(jpgpath));assert img is not None
 differences=[dict(frame=f,mae=float(np.abs(a[240:400,750:1280].astype(float)-img[240:400,750:1280]).mean())) for f,a in frames.items()];differences.sort(key=lambda x:x['mae'])
 image_alignment.append(dict(expected=frame,best=differences[:4],jpg_sha256=sha(jpgpath),jpg=jpgpath.relative_to(ROOT).as_posix()))
 draw=img.copy()
 for obj in row['labels']:
  identity=str(obj['id'])
  if identity not in ['00122657','00122746']:continue
  color=(0,0,255) if identity=='00122657' else (255,255,0);b=[round(obj['box2d'][k]) for k in ['x1','y1','x2','y2']];cv2.rectangle(draw,tuple(b[:2]),tuple(b[2:]),color,2);cv2.putText(draw,identity,tuple([b[0],b[1]-5]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
 cv2.imwrite(str(OUT/f'gt_{frame}.png'),cv2.resize(draw[220:430,650:1280],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
save('independent_audit.json',dict(passed=True,records=records,anchors=anchors,image_alignment=image_alignment,inputs={str(x.relative_to(ROOT)):sha(x) for x in [prev/'runtime.json',ROOT/ep['clock'],ROOT/ep['labels'],source,ROOT/ep['detector_file']]},scope='Independent archived matrix solve, exact GT coordinates and image-clock check; no models/tracker or production modifications. GT geometry is not assumed physically correct under occlusion.'))
print(json.dumps(dict(records=records,image_alignment=image_alignment),ensure_ascii=False))
