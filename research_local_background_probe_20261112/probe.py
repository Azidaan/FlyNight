from common import *
import gzip, cv2, numpy as np, lap
from scipy.optimize import linear_sum_assignment
from ultralytics.trackers.utils import matching
cv2.setNumThreads(2);cv2.setRNGSeed(0)
p=verify();v=p['video'];native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json')['videos'][v]
with np.load(ROOT/native['detections'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
with gzip.open(ROOT/('runs/domain_matched_native_20261109/rollouts/C34/'+v+'.events.json.gz'),'rt') as f:events=json.load(f)
e=next(x for x in events['events'] if x['frame']==539);target=next(i for i,r in enumerate(e['rows']) if r['tid']==232)
cap=cv2.VideoCapture(str(ROOT/native['video_path']));frames={}
for f in range(540):
 ok,img=cap.read();assert ok,f
 if f in p['frames']:frames[f]=img
cap.release()
def mask(f):
 h,w=frames[f].shape[:2];m=np.ones((h,w),np.uint8)*255;m[:20]=m[-20:]=0;m[:,:20]=m[:,-20:]=0
 for b in raw['xyxy'][(raw['frame_idx']==f)&(raw['conf']>=.1)]:
  x1,y1,x2,y2=b;cv2.rectangle(m,(int(np.floor(x1))-20,int(np.floor(y1))-20),(int(np.ceil(x2))+20,int(np.ceil(y2))+20),0,-1)
 cv2.rectangle(m,(900,475),(1100,620),0,-1);roi=np.zeros_like(m);roi[405:475,880:1135]=255;return cv2.bitwise_and(m,roi)
def transform(a,b):
 g0=cv2.cvtColor(frames[a],cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(frames[b],cv2.COLOR_BGR2GRAY);m0=mask(a);m1=mask(b)
 x=cv2.goodFeaturesToTrack(g0,maxCorners=2000,qualityLevel=.01,minDistance=4,mask=m0,blockSize=7)
 y,s,_=cv2.calcOpticalFlowPyrLK(g0,g1,x,None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01))
 back,t,_=cv2.calcOpticalFlowPyrLK(g1,g0,y,None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01))
 x=x[:,0];y=y[:,0];fb=np.linalg.norm(back[:,0]-x,axis=1);h,w=g0.shape
 valid=(s[:,0]>0)&(t[:,0]>0)&(fb<=1)&(y[:,0]>=0)&(y[:,0]<w)&(y[:,1]>=0)&(y[:,1]<h)
 idx=np.flatnonzero(valid);valid[idx]&=m1[y[idx,1].astype(int),y[idx,0].astype(int)]>0;x=x[valid];y=y[valid];fb=fb[valid]
 train=((np.floor(x[:,0]/20)+np.floor(x[:,1]/20)).astype(int)%2)==0
 delta=np.median(y[train]-x[train],axis=0);H=np.eye(3);H[:2,2]=delta;inliers=np.ones((int(train.sum()),1),np.uint8)
 pred=np.c_[x,np.ones(len(x))]@H[:2].T;err=np.linalg.norm(pred-y,axis=1);held=err[~train]
 center=np.array([998.1642700198172,548.8483609289349,1]);delta=(H@center-center)[:2]
 ready=int(train.sum())>=20 and int((~train).sum())>=20 and np.median(held)<=1 and np.quantile(held,.9)<=2
 return dict(frames=[a,b],H=H,train_count=int(train.sum()),heldout_count=int((~train).sum()),train_inliers=int(inliers.sum()),heldout_median=float(np.median(held)),heldout_p90=float(np.quantile(held,.9)),target_location_shift=delta,ready=bool(ready),points=x,destinations=y,train=train,errors=err,fb=fb)
trs=[transform(537,538),transform(538,539)]
boxes=[]
for r in e['rows']:
 m=np.array(r['mean'][:4]);boxes.append(np.r_[m[:2]-m[2:]/2,m[:2]+m[2:]/2])
boxes=np.array(boxes);ids=np.array(e['high_raw_ids']);det=raw['xyxy'][ids];conf=raw['conf'][ids]
# Reproduce the original detection adapter's float32 xyxy -> xywh -> tlwh -> xyxy.
converted=[]
for b in det:
 xywh=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]]
 tlwh=xywh.astype(np.float32).copy();tlwh[:2]-=tlwh[2:]/2
 xyxy=tlwh.copy();xyxy[2:]+=xyxy[:2];converted.append(xyxy)
det=np.array(converted)
def costs(bs):
 # Identical float32 geometry and score arithmetic to archived first assignment.
 c=matching.iou_distance(list(bs),list(det));return 1-(1-c)*conf[None,:]
base=costs(boxes);assert np.array_equal(base,np.array(e['C'],np.float32)),float(np.max(abs(base-np.array(e['C']))))
adjust=np.array(e['corrected'],np.float32)-base
def solve(C):
 _,x,y=lap.lapjv(C,extend_cost=True,cost_limit=.8);pairs=[[i,int(j)] for i,j in enumerate(x) if j>=0]
 # Independent augmented assignment: each unmatched endpoint costs0.4.
 n,m=C.shape;aug=np.zeros((n+m,n+m));aug[:n,:m]=C;aug[:n,m:]=.4;aug[n:,:m]=.4
 a,b=linear_sum_assignment(aug);obj=float(aug[a,b].sum());lapobj=sum(float(C[i,j]) for i,j in pairs)+.4*(n+m-2*len(pairs));assert abs(obj-lapobj)<1e-9
 return pairs,obj
basepairs,obj=solve(base+adjust);assert basepairs==e['actual_first']
def warp(bs,H):
 out=[]
 for x1,y1,x2,y2 in bs:
  q=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]])@H[:2].T;out.append(np.r_[q.min(axis=0),q.max(axis=0)])
 return np.array(out)
def arm(name,H,allrows):
 b=warp(boxes,H)
 if not allrows:b[np.arange(len(b))!=target]=boxes[np.arange(len(b))!=target]
 C=costs(b)+adjust;pairs,obj=solve(C);choice=next((j for i,j in pairs if i==target),None)
 return dict(name=name,H=H,allrows=allrows,target_box=b[target],target_min_cost=float(C[target].min()),target_best_raw_id=int(ids[C[target].argmin()]),selected_raw_id=None if choice is None else int(ids[choice]),pairs=pairs,assignments=[dict(tid=e['rows'][i]['tid'],raw_id=int(ids[j]),cost=float(C[i,j])) for i,j in pairs],matrix=C,objective=obj,boxes=b)
H=trs[1]['H'];residual=H@np.linalg.inv(trs[0]['H']);arms=[arm('identity',np.eye(3),False)]
for name,T in [('full',H),('residual',residual),('reverse',np.linalg.inv(H))]:
 for allrows in [False,True]:arms.append(arm(name+('_all' if allrows else '_232'),T,allrows))
# Empirical error sensitivity; samples are dependent diagnostics, not confidence bounds.
radius=max(t['heldout_p90'] for t in trs);sensitivity=[]
for name,T in [('full',H),('residual',residual)]:
 for dx in np.linspace(-radius,radius,5):
  for dy in np.linspace(-radius,radius,5):
   M=T.copy();M[0,2]+=dx;M[1,2]+=dy;q=arm(name+'_error',M,True)
   sensitivity.append(dict(mode=name,dx=float(dx),dy=float(dy),selected_raw_id=q['selected_raw_id'],assignments=q['assignments']))
verify();save('result.json',dict(passed=True,readiness_passed=all(t['ready'] for t in trs),registered=p['registered'],transforms=trs,arms=arms,sensitivity=sensitivity,error_radius=radius,row_ids=[r['tid'] for r in e['rows']],raw_ids=ids,frame_pixel_hashes={str(f):hashlib.sha256(img.tobytes()).hexdigest() for f,img in frames.items()},cv2_version=cv2.__version__,baseline_exact=True,input_unchanged=True,scope=p['limits']))
# A visual audit of point locations and excluded object; not an identity annotation.
for f in [538,539]:
 img=frames[f].copy();cv2.rectangle(img,(900,475),(1100,620),(0,0,255),2);cv2.rectangle(img,(880,405),(1135,475),(255,255,0),2)
 t=trs[1]
 for q,train,err in zip(t['points'] if f==538 else t['destinations'],t['train'],t['errors']):
  if not train:cv2.circle(img,tuple(np.rint(q).astype(int)),2,(0,255,0) if err<=2 else (0,140,255),-1)
 cv2.imwrite(str(OUT/('background_'+str(f)+'.jpg')),img)
print(json.dumps(dict(ready=all(t['ready'] for t in trs),transforms=[{k:v for k,v in t.items() if k in ['frames','train_count','heldout_count','heldout_median','heldout_p90','ready']} for t in trs],arms=[{k:a[k] for k in ['name','target_min_cost','selected_raw_id']} for a in arms]),ensure_ascii=False))
