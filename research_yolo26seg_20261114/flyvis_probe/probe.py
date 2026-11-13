"""Frozen saved-map CPU mask probe; evaluate labels only after predictions saved."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path('E:/Dissertation');OUT=Path(__file__).resolve().parent
for v in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR'):
    q=OUT/'tmp';q.mkdir(parents=True,exist_ok=True);os.environ[v]=str(q)
import hashlib,json,math,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import cv2
from scipy.spatial import cKDTree,ConvexHull
import psutil

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    with (OUT/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_np(name,**arrays):
    with (OUT/name).open('xb') as f:np.savez_compressed(f,**arrays)
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()

start=time.perf_counter();cv2.setNumThreads(2)
p=read(OUT/'protocol.json')
assert (OUT/'GO.json').exists(),'Parent must confirm protocol/card registration before calculation'
go=read(OUT/'GO.json');assert go['protocol_sha256']==sha(OUT/'protocol.json')
reserve=max(6*1024**3,.2*psutil.virtual_memory().total)
assert psutil.virtual_memory().available>reserve+2*1024**3,'Insufficient free system memory'
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
points=np.asarray(g['points'],int);assert points.shape==(721,2) and g['eye_size']==391
generated=np.asarray([(int(13*v)+195,int(13*(u+v/2))+195)
    for u in range(-15,16) for v in range(max(-15,-15-u),min(15,-u+15)+1)])
assert np.array_equal(points,generated),'Archived point order differs from BoxEye'
px,py=points.T
grid=np.stack(np.meshgrid(np.arange(391),np.arange(391)),axis=-1)
nearest=cKDTree(points).query(grid.reshape(-1,2),k=1)[1].reshape(391,391)
hull=ConvexHull(points)
inside=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),points[hull.vertices].astype(np.int32),1).astype(bool)
geometry=[];features={};raw_frames={}

def eye_sample(image):
    im=cv2.resize(image.astype(np.float32),(391,391),interpolation=cv2.INTER_AREA)
    blurred=cv2.boxFilter(im,-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)
    value=blurred[py,px]
    return value[:,None] if value.ndim==1 else value

for record in p['records']:
    frame=record['frame'];x1,y1,x2,y2=record['crop'];w=x2-x1;h=y2-y1
    image=cv2.imread(str(ROOT/f'research_visible_part_2179_20261114/frames/{frame}.png'))
    assert image.shape==(720,1280,3);raw_frames[frame]=image
    c=np.load(ROOT/record['map'],allow_pickle=False)
    assert c.shape==(721,34) and c.dtype==np.float32 and np.isfinite(c).all()
    retinal=np.stack([np.load(ROOT/r['path'],allow_pickle=False) for r in record['retinal']]).astype(np.float32)
    assert retinal.shape==(len(record['sequence']),721) and np.isfinite(retinal).all()
    # Same history and recurrence as receptor_features_rich, without module import.
    taus=np.geomspace(1.5,32.0,17);ema=np.repeat(retinal[:1],17,axis=0).astype(np.float64)
    for current in retinal:ema+=(current[None,:]-ema)/taus[:,None]
    history=np.concatenate([ema,current[None,:]-ema],axis=0).T
    rgb=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)/255.
    gray=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)/255.
    features[frame]={'FlyVis34':c.astype(np.float64),'Retina_history34':history,
        'Retina_current1':retinal[-1,:,None].astype(np.float64),
        'Raw_gray1':eye_sample(gray).astype(np.float64),'Raw_RGB3':eye_sample(rgb).astype(np.float64)}
    # Exact resize coordinate transformation; no coordinates become model input.
    centers=np.column_stack([x1+(px+.5)*w/391-.5,y1+(py+.5)*h/391-.5])
    roundtrip=np.column_stack([(centers[:,0]-x1+.5)*391/w-.5,(centers[:,1]-y1+.5)*391/h-.5])
    error=float(abs(roundtrip-points).max());assert error<1e-10
    geometry.append(dict(frame=frame,crop=record['crop'],eye_size=391,receptors=721,
        frame_center_range=[centers.min(0).tolist(),centers.max(0).tolist()],
        roundtrip_error=error,cell_window_source_pixels=[13*w/391,13*h/391]))

masks_path=ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz'
label_access=[]
def known_labels(frame):
    label_access.append(dict(frame=frame,time=now()))
    with np.load(masks_path,allow_pickle=False) as masks:
        target=masks[f'{frame}_target'].astype(bool);neighbor=masks[f'{frame}_neighbor'].astype(bool)
        other=masks[f'{frame}_background'].astype(bool)|masks[f'{frame}_occluder'].astype(bool)
        uncertain=masks[f'{frame}_uncertain'].astype(bool)
    assert np.all(target.astype(int)+neighbor.astype(int)+other.astype(int)+uncertain.astype(int)==1)
    return [other,target,neighbor],uncertain

# Only the training member is opened before models and predictions are fixed.
known,_=known_labels(1159)
train_rec=next(r for r in p['records'] if r['frame']==1159)
x1,y1,x2,y2=train_rec['crop'];w=x2-x1;h=y2-y1
coverage=np.stack([eye_sample(m[y1:y2,x1:x2].astype(np.float32))[:,0] for m in known],axis=1)
accepted=coverage.max(axis=1)>=.9
y=coverage.argmax(axis=1)
counts=np.bincount(y[accepted],minlength=3)

# Independently reconstruct each footprint from actual separable resize weights.
# This checks every selected cell against scalar pixel contributions, including
# partial input pixels and padding. It does not assume old edge-based to_frame.
rx=cv2.resize(np.eye(w,dtype=np.float32),(391,w),interpolation=cv2.INTER_AREA).T.astype(np.float64)
ry=cv2.resize(np.eye(h,dtype=np.float32),(391,h),interpolation=cv2.INTER_AREA).T.astype(np.float64)
independent=np.zeros_like(coverage,dtype=np.float64)
for i,(xx,yy) in enumerate(points):
    wx=rx[max(0,xx-6):min(391,xx+7)].sum(0)/13
    wy=ry[max(0,yy-6):min(391,yy+7)].sum(0)/13
    ix=np.flatnonzero(wx);iy=np.flatnonzero(wy)
    for cls,m in enumerate(known):
        patch=m[y1:y2,x1:x2]
        independent[i,cls]=math.fsum(float(wy[j]*wx[k])*int(patch[j,k]) for j in iy for k in ix)
mapping_error=float(abs(independent-coverage).max())
save('input_geometry_audit.json',dict(passed=mapping_error<1e-6,frames=geometry,
    receptor_order_exact=True,independent_pixel_footprint_max_error=mapping_error,
    training_known_cells_by_class=counts.tolist(),training_unknown_cells=int((~accepted).sum()),
    coordinate_features_used=False,source='BoxEye._receptor_centers / SparseViews.render; pixel-centre inverse and exact INTER_AREA impulse weights.'))
assert mapping_error<1e-6,('Geometry footprint failed',mapping_error)
if np.min(counts)<3:
    save('result.json',dict(passed=False,reason='Fewer than3 training cells in one class',counts=counts.tolist(),new_core_calls=0))
    raise SystemExit('INSUFFICIENT_TRAIN_CELL_SUPPORT')

models={};fit_seconds={};pred={};prediction_seconds={};coeff={}
for arm in p['arms']:
    t=time.perf_counter();x=features[1159][arm][accepted];lab=y[accepted]
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.;z=(x-mu)/sd
    weights=1/(3*counts[lab]);weights=weights/weights.sum()
    onehot=np.eye(3)[lab];zm=weights@z;ym=weights@onehot
    zx=z-zm;yx=onehot-ym
    coef=np.linalg.solve(zx.T@(zx*weights[:,None])+np.eye(z.shape[1]),zx.T@(yx*weights[:,None]))
    intercept=ym-zm@coef;models[arm]=(mu,sd,coef,intercept)
    fit_seconds[arm]=time.perf_counter()-t
    for name,array in zip(['mu','sd','coef','intercept'],models[arm]):coeff[arm+'_'+name]=array
    for frame in (1159,1164,1165):
        t=time.perf_counter();score=((features[frame][arm]-mu)/sd)@coef+intercept
        assert score.shape==(721,3) and np.isfinite(score).all()
        labels=score.argmax(axis=1);raster=labels[nearest].astype(np.int8);raster[~inside]=-1
        pred[f'{arm}_{frame}_score']=score;pred[f'{arm}_{frame}_cells']=labels
        pred[f'{arm}_{frame}_raster']=raster
        prediction_seconds[f'{arm}_{frame}']=time.perf_counter()-t
save_np('models.npz',**coeff)
save_np('predictions.npz',**pred)
prediction_saved=now()
save('prediction_receipt.json',dict(created=prediction_saved,
    inputs_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),
    models_sha256=sha(OUT/'models.npz'),predictions_sha256=sha(OUT/'predictions.npz'),
    label_reads_before_predictions=label_access.copy(),eval_labels_read=False,
    no_coordinates_or_evaluation_labels_in_model=True,new_core_calls=0,gpu_calls=0))

def metric(true,prediction):
    valid=true>=0;cm=np.zeros((3,3),int)
    for actual,estimate in zip(true[valid].ravel(),prediction[valid].ravel()):cm[actual,estimate]+=1
    rows=[]
    for cls in range(3):
        tp=int(cm[cls,cls]);fn=int(cm[cls].sum()-tp);fp=int(cm[:,cls].sum()-tp)
        rows.append(dict(class_index=cls,tp=tp,fn=fn,fp=fp,iou=tp/(tp+fp+fn) if tp+fp+fn else None,
            recall=tp/(tp+fn) if tp+fn else None))
    return dict(confusion=cm.tolist(),classes=rows,car_mean_iou=float(np.mean([rows[1]['iou'],rows[2]['iou']])),valid_pixels=int(valid.sum()))

results=[];evaluation_labels={};counts_report=[]
for rec in p['records']:
    frame=rec['frame'];x1,y1,x2,y2=rec['crop'];w=x2-x1;h=y2-y1
    classes,uncertain=known_labels(frame)
    # Native pixel labels; prediction is projected from nearest grid cells.
    ys=(np.arange(h)+.5)*391/h-.5;xs=(np.arange(w)+.5)*391/w-.5
    eye_grid=np.stack(np.meshgrid(xs,ys),axis=-1)
    d,idx=cKDTree(points).query(eye_grid.reshape(-1,2),k=1)
    in_eye=(eye_grid.reshape(-1,2)@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9).all(axis=1).reshape(h,w)
    truth=np.full((h,w),-1,np.int8)
    for cls,mask in enumerate(classes):truth[mask[y1:y2,x1:x2]&in_eye]=cls
    evaluation_labels[frame]=truth
    cc=np.stack([eye_sample(m[y1:y2,x1:x2].astype(np.float32))[:,0] for m in classes],axis=1)
    cell_valid=cc.max(1)>=.9;cell_class=cc.argmax(1)
    counts_report.append(dict(frame=frame,role=rec['role'],
        cells_by_class=np.bincount(cell_class[cell_valid],minlength=3).tolist(),
        unknown_cells=int((~cell_valid).sum()),pixels_by_class=[int((truth==k).sum()) for k in range(3)],
        confident_core_pixels_outside_eye=[int((m[y1:y2,x1:x2]&~in_eye).sum()) for m in classes[1:]],
        total_core_pixels_whole_frame=[int(m.sum()) for m in classes[1:]],
        core_pixels_in_crop=[int(m[y1:y2,x1:x2].sum()) for m in classes[1:]]))
    for arm in p['arms']:
        raster=pred[f'{arm}_{frame}_cells'][idx].reshape(h,w)
        results.append(dict(arm=arm,frame=frame,role=rec['role'],**metric(truth,raster)))

for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
process=psutil.Process();rss=process.memory_info().rss
scope='Frozen archived neural maps; first-frame supervised three-class per-cell readout. Same-scene same-object transfer, not independent-video generalization or full silhouette ground truth.'
result=dict(passed=True,created=now(),protocol_sha256=sha(OUT/'protocol.json'),
    source_sha256=sha(__file__),prediction_receipt_sha256=sha(OUT/'prediction_receipt.json'),
    counts=counts_report,scores=results,fit_seconds=fit_seconds,prediction_seconds=prediction_seconds,
    total_seconds=time.perf_counter()-start,rss_bytes=rss,
    peak_working_set_bytes=getattr(process.memory_info(),'peak_wset',None),
    label_access_log=label_access,scope=scope,new_core_calls=0,gpu_calls=0,full_tracker_calls=0,
    original_maps_unchanged=True,feature_dimensions={arm:features[1159][arm].shape[1] for arm in p['arms']},
    warnings=['Onlyknownpixelsinsideeyeareevaluated;uncertainfullsilhouettenotgroundtruth.',
        'Featuresretainarchivedmovingcropandmissing1160historystep;notnewcontinuousvideoeye.',
        'Matchedhistory34controlhassamefeaturecount;RGB/graycontrolshavefewerfeaturesbutsamefixedlearner.',
        'OldretinalfeaturesincludeLamina preprocessing;rawRGBandgraycontrolsuseoriginalpixels.'])
save('result.json',result)
print('FLYVIS_MASK_CPU_COMPLETE',json.dumps([r for r in results if r['role']=='evaluation'],ensure_ascii=False))
