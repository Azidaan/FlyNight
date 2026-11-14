from common import *
import cv2,numpy as np
from scipy.spatial import cKDTree,ConvexHull
db=read(OUT/'dataset.json');result=read(OUT/'result.json');arr=np.load(OUT/'dataset.npz');model=np.load(OUT/'models.npz');pred=np.load(OUT/'predictions.npz');points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);yy,xx=np.indices((391,391));near=cKDTree(points).query(np.c_[xx.ravel(),yy.ravel()])[1].reshape(391,391);sup=cv2.fillConvexPoly(np.zeros((391,391),'uint8'),points[ConvexHull(points).vertices],1).astype(bool);mx=0.;dx=0
for r in result['records']:
 key=r['request'];truth=arr[key+'__truth'];inside=arr[key+'__inside'];k=(truth>=0)&sup
 if r['arm']=='Box':mask=inside
 else:
  pre=f"{r['mode']}__{r['arm']}__{r['seed']}";x=((arr[key+'__'+r['arm']]-model[pre+'__mu'])/model[pre+'__sd']);h=np.maximum(x@model[pre+'__0.weight'].T+model[pre+'__0.bias'],0);a=h@model[pre+'__2.weight'].T+model[pre+'__2.bias'];orig=pred[pre+'__'+key];mx=max(mx,float(np.max(abs(a-orig))));assert np.array_equal(a.argmax(1),orig.argmax(1));mask=a.argmax(1)[near]==1;dx+=1
 tn,fp,fn,tp=map(int,np.bincount(2*(truth[k]==1).astype(int)+mask[k],minlength=4));assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn']);assert np.count_nonzero(mask[k&inside&(truth!=1)])==r['inside_fp']
assert mx<1e-4;save('audit.json',dict(passed=True,matrices=len(result['records']),reader_predictions_recreated=dx,max_logit_error=mx,train_val_disjoint=True,sha256=sha(OUT/'result.json')))
