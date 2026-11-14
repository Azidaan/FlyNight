from common import *
import numpy as np,cv2
from scipy.spatial import cKDTree,ConvexHull
PTS=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);PX,PY=PTS.T
NEAR=cKDTree(PTS).query(PTS,k=7)[1]
GRID=np.stack(np.meshgrid(np.arange(391),np.arange(391)),axis=-1)
NEAREST=cKDTree(PTS).query(GRID.reshape(-1,2))[1].reshape(391,391)
SUPPORT=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),PTS[ConvexHull(PTS).vertices],1).astype(bool)
def sample(a):return cv2.boxFilter(cv2.resize(a.astype(np.float32),(391,391),interpolation=cv2.INTER_AREA),-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)[PY,PX]
def load_record(r):
    x1,y1,x2,y2=r['crop'];im=cv2.imread(str(ROOT/r['image']));assert im is not None;patch=cv2.resize(im[y1:y2,x1:x2],(391,391))
    if 'mask' in r:
        z=np.load(ROOT/r['mask']);pos=z[r['keys'][1]].astype(bool);neg=z[r['keys'][2]].astype(bool)
        bg=z[r['keys'][0]].astype(bool)|z[r['occluder']].astype(bool) if r['keys'][0] else ~(pos|neg|z[r['uncertain']].astype(bool));raw=[a[y1:y2,x1:x2] for a in [bg,pos,neg]]
        rasters=[cv2.resize(a.astype('uint8'),(391,391),interpolation=cv2.INTER_NEAREST).astype(bool) for a in raw]
    else:
        raw=[cv2.fillPoly(np.zeros((391,391),np.uint8),[np.array(r['polygons'][key],np.int32)],1).astype(bool) for key in ['background','target','neighbor']];rasters=raw
    assert (np.stack(rasters).sum(0)<=1).all(),r['id'];coverage=np.stack([sample(a) for a in raw],1);labels=np.where(coverage.max(1)>=.9,coverage.argmax(1),-1);truth=np.full((391,391),-1,dtype=np.int8)
    for k,a in enumerate(rasters):truth[a]=k
    fly=np.load(ROOT/r['map']).astype(float);history=np.stack([np.load(ROOT/t['path']) for t in r['retinal']]).astype(float);ema=np.repeat(history[:1],17,axis=0)
    for cur in history:ema+=(cur-ema)/np.geomspace(1.5,32,17)[:,None]
    retina=np.concatenate((ema,cur-ema)).T;rgb=sample(im[y1:y2,x1:x2]/255.0)
    box=np.array(r['box']);center=(box[:2]+box[2:])/2;xy=np.c_[x1+(PX+.5)*(x2-x1)/391-.5,y1+(PY+.5)*(y2-y1)/391-.5];g=(xy-center)/(box[2:]-box[:2]);f=np.c_[fly,fly[NEAR].mean(1)];v=np.c_[retina,retina[NEAR].mean(1)]
    features={'FlyVis+G':np.c_[f,g],'Retina+G':np.c_[v,g],'Geometry':g,'RGB+G':np.c_[rgb,rgb[NEAR].mean(1),g],'FlyVis':f,'Retina':v}
    assert all(np.isfinite(a).all() for a in features.values())
    gx=(np.arange(391)+.5)*(x2-x1)/391+x1-.5;gy=(np.arange(391)+.5)*(y2-y1)/391+y1-.5;inside=(gx[None]>=box[0])&(gx[None]<=box[2])&(gy[:,None]>=box[1])&(gy[:,None]<=box[3])
    return dict(features=features,cell_labels=labels,truth=truth,image=patch,inside=inside)
def measure(labels,d):
    raster=labels[NEAREST];known=(d['truth']>=0)&SUPPORT;truth=d['truth'];pred=raster==1;tp=int((known&pred&(truth==1)).sum());fp=int((known&pred&(truth!=1)).sum());fn=int((known&~pred&(truth==1)).sum());negative=known&d['inside']&(truth!=1);n=int(negative.sum());fpi=int((negative&pred).sum())
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),inside_negative_pixels=n,inside_false_positive=fpi,inside_fpr=fpi/max(1,n)),raster
