from common import *
import numpy as np
from scipy.spatial import cKDTree
import cv2
p=read(OUT/'protocol_v2.json');res=read(OUT/'result.json');pred=np.load(OUT/'predictions.npz');data=np.load(OUT/'evaluation_arrays.npz');weights=np.load(OUT/'models.npz')
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
assert res['source_unchanged'] and not res['test_labels_used_in_fitting']
assert {r['video'] for r in p['records'] if r['split']=='train'}.isdisjoint({r['video'] for r in p['records'] if r['split']=='test'})
pts=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points']);neighbors=cKDTree(pts).query(pts,k=7)[1]
nearest=cKDTree(pts).query(np.stack(np.meshgrid(np.arange(391),np.arange(391)),axis=-1).reshape(-1,2))[1].reshape(391,391)
checks=[];maxerr=0.;overlaps={}
for r in p['records']:
    x1,y1,x2,y2=r['crop'];box=np.array(r['box']);center=(box[:2]+box[2:])/2;wh=box[2:]-box[:2]
    position=np.c_[x1+(pts[:,0]+.5)*(x2-x1)/391-.5,y1+(pts[:,1]+.5)*(y2-y1)/391-.5];geom=(position-center)/wh
    history=np.stack([np.load(ROOT/t['path']) for t in r['retinal']]).astype(float);av=np.repeat(history[:1],17,axis=0)
    for current in history:av+=(current-av)/np.geomspace(1.5,32,17)[:,None]
    retinal=np.concatenate((av,current-av)).T;fly=np.load(ROOT/r['map']).astype(float)
    for arm,raw in [('FlyVis',fly),('Control',retinal),('Geometry',None)]:
        feature=geom if raw is None else np.concatenate((raw,raw[neighbors].mean(1),geom),axis=1)
        assert np.isfinite(feature).all()
        x=((feature-weights[arm+'_mu'])/weights[arm+'_sd']).astype('float32');ss=[]
        for seed in [42,43,44]:
            pref=f'{arm}_{seed}_';h=np.maximum(0,x@weights[pref+'0.weight'].T+weights[pref+'0.bias']);s=h@weights[pref+'2.weight'].T+weights[pref+'2.bias'];s-=s.max(axis=1,keepdims=True);ss.append(np.exp(s)/np.exp(s).sum(axis=1,keepdims=True))
        score=np.mean(ss,axis=0);given=pred[arm+'__'+r['id']];assert np.isfinite(given).all();err=float(np.max(np.abs(score-given)));maxerr=max(maxerr,err);assert err<1e-5
        if r['split']=='train':continue
        lab=data[arm+'__'+r['id']];assert np.array_equal(lab,given.argmax(axis=1)[nearest]);truth=data['truth__'+r['id']];known=data['support']&(truth>=0)
        cm=np.bincount((truth[known]*3+lab[known]).ravel(),minlength=9).reshape(3,3);tp=int(cm[1,1]);fp=int(cm[:,1].sum()-tp);fn=int(cm[1].sum()-tp)
        xs=x1+(np.arange(391)+.5)*(x2-x1)/391-.5;ys=y1+(np.arange(391)+.5)*(y2-y1)/391-.5;inside=(xs[None,:]>=box[0])&(xs[None,:]<=box[2])&(ys[:,None]>=box[1])&(ys[:,None]<=box[3]);negative=inside&known&(truth!=1);fpinside=int(np.count_nonzero(negative&(lab==1)));n=int(negative.sum())
        row=next(v for v in res['records'] if v['id']==r['id'] and v['arm']==arm)
        for k,v in dict(tp=tp,fp=fp,fn=fn,inside_negative_pixels=n,inside_false_positive=fpinside).items():assert row[k]==v
        for k,v in dict(iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),inside_fpr=fpinside/max(1,n)).items():assert abs(row[k]-v)<1e-12
        checks.append(dict(id=r['id'],arm=arm,confusion_matrix=cm.tolist()))
    if 'polygons' in r:
        masks=[cv2.fillPoly(np.zeros((391,391),np.uint8),[np.array(poly,np.int32)],1).astype(bool) for poly in r['polygons'].values()];overlap=int((np.stack(masks).sum(0)>1).sum());overlaps[r['id']]=overlap;assert overlap==0
new=[r for r in res['records'] if r['split']=='test'];means={arm:float(np.mean([r['iou'] for r in new if r['arm']==arm])) for arm in ['FlyVis','Control','Geometry']};wins=sum(next(r['iou'] for r in new if r['id']==v['id'] and r['arm']=='FlyVis')>v['iou'] for v in new if v['arm']=='Control');quality=all(r['recall']>=.5 and r['inside_fpr']<=.1 for r in new if r['arm']=='FlyVis');increment=means['FlyVis']>means['Control'] and wins>=2
save('summary.json',dict(scientific_passed=quality and increment,quality_passed=quality,increment_passed=increment,new_video_mean_iou=means,new_video_wins=wins,scope='Three manually annotated visible-core image tests from unseen training videos; one seed ensemble, not identity tracking or complete segmentation.',limitations=['Only four training frames from three videos.','Conservative manual cores omit unknown boundaries; apparent geometry IoU=1 on one case is not a full contour score.','Old1164 uses1165query ROI, so diagnostic only; old1165 has same-video training1159.','No reference appearance vector conditions target identity: classes rely on ROI location and observed features.','This compound reader change does not isolate individual architecture or training-label effects.']))
save('audit.json',dict(passed=True,metric_tables=checks,prediction_max_abs_error=maxerr,annotation_overlap_pixels=overlaps,source_unchanged=True,artifact_hashes={n:sha(OUT/n) for n in ['result.json','models.npz','predictions.npz','evaluation_arrays.npz','summary.json']}))
print('AUDIT_PASSED',maxerr,means,'gate',quality and increment)
