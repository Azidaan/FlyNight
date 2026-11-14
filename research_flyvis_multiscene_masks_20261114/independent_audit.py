from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,time
from scipy.spatial import ConvexHull
p=read(OUT/'protocol.json');fit=read(OUT/'fit_result.json');result=read(OUT/'result.json');review=read(OUT/'teacher_review.json');teacher=read(OUT/review['teacher_result'])
features=np.load(OUT/'features.npz',allow_pickle=False);labels=np.load(OUT/review['teacher_masks'],allow_pickle=False);models=np.load(OUT/'models.npz',allow_pickle=False);pred=np.load(OUT/'predictions.npz',allow_pickle=False)
points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
verified=[];ys={};cs={}
for r in p['records']:
 if r['split']!='train':continue
 row=next(t for t in teacher['records'] if t['id']==r['id'])
 if not row['automatic_gate'] or r['id'] in review['exclude_ids']:continue
 image=cv2.resize(labels[r['id']].astype(np.float32),(391,391),interpolation=cv2.INTER_AREA);pad=np.pad(image,6);coverage=np.array([np.sum(pad[y:y+13,x:x+13],dtype=np.float64)/169 for x,y in points]);y=np.where(coverage>=.9,1,np.where(coverage<=.1,0,-1));c=np.array([(y==0).sum(),(y==1).sum()])
 if c.min()<3:continue
 verified.append(r);ys[r['id']]=y;cs[r['id']]=c
assert [r['id'] for r in verified]==fit['accepted']
videos=sorted({r['video'] for r in verified});assert set(videos).isdisjoint(p['test_videos']) and len(videos)>=8
errors=[];score_errors=[]
for size,ids in [('small',fit['small']),('all',fit['accepted'])]:
 rows=[r for r in verified if r['id'] in ids]
 for arm in p['arms']:
  xs=[];labs=[];weights=[]
  for r in rows:
   y=ys[r['id']];valid=y!=-1;xs.append(features[r['id']+'__'+arm][valid].astype(np.float64));labs.extend(y[valid].tolist());nf=sum(t['video']==r['video'] for t in rows);weights.extend([1/(len(videos)*nf*2*cs[r['id']][k]) for k in y[valid]])
  x=np.concatenate(xs);y=np.array(labs);w=np.array(weights);mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;design=np.c_[(x-mu)/sd,np.ones(len(x))];penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
  solution=np.linalg.lstsq(design.T@(design*w[:,None])+penalty,design.T@(np.eye(2)[y]*w[:,None]),rcond=None)[0];key=size+'__'+arm
  err=float(abs(solution-np.r_[models[key+'__coef'],models[key+'__intercept'][None]]).max());errors.append(dict(size=size,arm=arm,max_coefficient_error=err));assert err<1e-9
  for r in p['records']:
   scores=np.c_[(features[r['id']+'__'+arm].astype(float)-mu)/sd,np.ones(721)]@solution;delta=float(abs(scores-pred[key+'__'+r['id']]).max());score_errors.append(delta);assert delta<1e-8
manual=np.load(OUT/'manual_test_masks.npz',allow_pickle=False);nearest=np.empty((391,391),int)
for y in range(391):
 d=(np.arange(391)[:,None]-points[None,:,0])**2+(y-points[None,:,1])**2;nearest[y]=d.argmin(axis=1)
# Equidistant cells are possible; audit metrics using saved rasters and verify
# their score label is one of the true nearest cells at each location.
rasters=np.load(OUT/'raster_predictions.npz',allow_pickle=False);inside=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),points[ConvexHull(points).vertices],1).astype(bool)
metric_checks=[]
for r in result['records']:
 key=r['size']+'__'+r['arm']+'__'+r['id'];raster=rasters[key].astype(int);cells=pred[key].argmax(1);reference=cells[nearest];different=reference!=raster
 for y,x in zip(*np.where(different)):
  ds=np.sum((points-np.array([x,y]))**2,axis=1);assert raster[y,x] in cells[ds==ds.min()]
 pos=manual[r['id']+'__positive'].astype(bool);neg=manual[r['id']+'__negative'].astype(bool);known=(pos|neg)&inside;truth=pos.astype(int);cm=np.bincount((2*truth[known]+raster[known]).ravel(),minlength=4)
 tn,fp,fn,tp=map(int,cm);assert [tp,fp,fn,tn]==[r[k] for k in ['tp','fp','fn','tn']];assert abs(tp/max(tp+fp+fn,1)-r['iou'])<1e-12;metric_checks.append(key)
unchanged={n:sha(ROOT/n)==h for n,h in p['inputs'].items()};assert all(unchanged.values())
assert not fit['manual_test_labels_opened'];assert read(OUT/'fit_registration.json')['inputs'][rel(OUT/'manual_test_masks.npz')]==sha(OUT/'manual_test_masks.npz')
save('audit.json',dict(passed=True,created=now(),coefficient_checks=errors,max_prediction_error=max(score_errors),confusion_tables_recomputed=len(metric_checks),nearest_tie_policy_validated=True,source_checks=len(unchanged),all_sources_unchanged=True,train_test_videos_disjoint=True,training_labels_independently_sampled=True,alignment=read(OUT/'alignment_audit.json')['passed'],limitations='Evaluator shares stored features, teacher and manual labels, and cv2 resize/hull with primary. Solver uses separate augmented least-squares system, metrics use independent bincount; it is not an independent human annotation.',source_sha256=sha(__file__)))
print('INDEPENDENT_AUDIT_PASSED',len(metric_checks),max(score_errors))
