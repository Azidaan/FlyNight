from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,time,psutil
from scipy.spatial import cKDTree,ConvexHull
start=time.perf_counter();p=read(OUT/'protocol.json');gate=read(OUT/'fit_registration.json')
for n,h in gate['inputs'].items():assert sha(ROOT/n)==h,n
assert read(OUT/'alignment_audit.json')['passed'];review=read(OUT/'teacher_review.json');teacher=read(OUT/review['teacher_result']);points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);px,py=points.T
features=np.load(OUT/'features.npz',allow_pickle=False);masks=np.load(OUT/review['teacher_masks'],allow_pickle=False)
accepted=[];exclusions=[];labels={};counts={}
for t in teacher['records']:
 r=next(r for r in p['records'] if r['id']==t['id']);assert r['split']=='train'
 if not t['automatic_gate'] or r['id'] in review['exclude_ids']:
  exclusions.append(dict(id=r['id'],reason='teacher automatic gate' if not t['automatic_gate'] else review['exclude_ids'][r['id']]));continue
 mask=cv2.resize(masks[r['id']].astype(np.float32),(391,391),interpolation=cv2.INTER_AREA);coverage=cv2.boxFilter(mask,-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)[py,px]
 y=np.full(721,-1,int);y[coverage>=.9]=1;y[coverage<=.1]=0;c=np.bincount(y[y>=0],minlength=2)
 if c.min()<3:exclusions.append(dict(id=r['id'],reason='fewer than3 clear cells in a class',counts=c.tolist()));continue
 labels[r['id']]=y;counts[r['id']]=c;accepted.append(r)
videos=sorted({r['video'] for r in accepted});assert len(videos)>=8
small=[min([r for r in accepted if r['video']==v],key=lambda r:r['frame']) for v in videos]
assert not set(videos)&set(p['test_videos'])
models={};predictions={};training=[];fit_times={}
for size,rows in [('small',small),('all',accepted)]:
 for arm in p['arms']:
  st=time.perf_counter();xx=[];yy=[];ww=[]
  for r in rows:
   y=labels[r['id']];valid=y>=0;xx.append(features[r['id']+'__'+arm][valid].astype(float));yy.append(y[valid]);nframes=sum(s['video']==r['video'] for s in rows)
   ww.append(1/(len(videos)*nframes*2*counts[r['id']][y[valid]]))
  x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);assert abs(w.sum()-1)<1e-10
  mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;z=(x-mu)/sd;onehot=np.eye(2)[y];zm=w@z;ym=w@onehot;zx=z-zm;yx=onehot-ym
  coef=np.linalg.solve(zx.T@(zx*w[:,None])+np.eye(z.shape[1]),zx.T@(yx*w[:,None]));intercept=ym-zm@coef
  key=size+'__'+arm
  for name,a in [('mu',mu),('sd',sd),('coef',coef),('intercept',intercept)]:models[key+'__'+name]=a
  for r in p['records']:
   score=((features[r['id']+'__'+arm].astype(float)-mu)/sd)@coef+intercept;assert np.isfinite(score).all();predictions[key+'__'+r['id']]=score
  fit_times[key]=time.perf_counter()-st
  pred=z@coef+intercept;training.append(dict(size=size,arm=arm,frames=len(rows),videos=len(videos),cells=len(y),weighted_accuracy=float(w@(pred.argmax(1)==y))))
with (OUT/'models.npz').open('xb') as f:np.savez_compressed(f,**models)
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**predictions)
save('fit_result.json',dict(created=now(),accepted=[r['id'] for r in accepted],small=[r['id'] for r in small],excluded=exclusions,training=training,fit_seconds=fit_times,seconds=time.perf_counter()-start,rss_bytes=psutil.Process().memory_info().rss,models_sha256=sha(OUT/'models.npz'),predictions_sha256=sha(OUT/'predictions.npz'),source_sha256=sha(__file__),manual_test_labels_opened=False,test_videos=p['test_videos'],training_videos=videos,full_tracker_calls=0,core_updates=0,random_seed=None))
print('FIT_COMPLETE',len(accepted),len(videos),flush=True)
