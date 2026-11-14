from official import *
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
import torch,time
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');cases=read(OUT/'cases.json')['cases'];lab=zipfile.ZipFile(ROOT/p['labelzip']);cache={};rows=[];times=[];raws={}
def run(model,im,classes,dx,dy,tile):
 torch.cuda.synchronize();start=time.perf_counter();d=model.predict(im,imgsz=960,conf=.01,iou=.7,classes=classes,max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];torch.cuda.synchronize();secs=time.perf_counter()-start;return [dict(box=(box+np.array([dx,dy,dx,dy])).tolist(),score=float(conf),cls=int(cl),tile=tile) for box,conf,cl in zip(d.boxes.xyxy.cpu().numpy(),d.boxes.conf.cpu().numpy(),d.boxes.cls.cpu().numpy())],secs
def assign(items,truth,fixed):
 rows=[i for i,x in enumerate(fixed) if x is None];result=list(fixed)
 if not rows or not items:return result
 mat=np.array([[iou(truth[i],np.array(x['box'])) for x in items] for i in rows]);avail=np.array([all(a is None or iou(np.array(a['box']),np.array(x['box']))<.7 for a in fixed) for x in items]);obj=np.where((mat>=.5)&avail[None],1+mat,0);ii,jj=linear_sum_assignment(-np.c_[obj,np.zeros((len(rows),len(rows)))])
 for i,j in zip(ii,jj):
  if j<len(items) and obj[i,j]>0:result[rows[i]]=dict(items[j],index=int(j),overlap=float(mat[i,j]))
 return result
def diag(items,bb):
 if not items:return None
 vals=[iou(bb,np.array(a['box'])) for a in items];i=int(np.argmax(vals));return dict(items[i],iou=float(vals[i]))
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for c in cases:
  cid,v,idx=c['id'],c['video'],c['index'];im=cv2.imread(str(ROOT/c['frames'][0]['image']));assert im.shape[:2]==(720,1280)
  if v not in cache:cache[v]=labels(lab,v)
  frame=cache[v][idx];objects=[next(o for o in frame['labels'] if o['id']==i) for i in c['ids']];truth=[bbox(o) for o in objects];full,tf=run(model,im,[2],0,0,-1);fullall,ta=run(model,im,None,0,0,-1);tile=[];tileall=[];tt=0;tta=0
  for t,(x,y) in enumerate([(0,0),(512,0),(0,272),(512,272)]):
   patch=im[y:y+448,x:x+768];a,sec=run(model,patch,[2],x,y,t);tile.extend(a);tt+=sec;b,sec=run(model,patch,None,x,y,t);tileall.extend(b);tta+=sec
  fixed=assign(full,truth,[None,None]);combined=assign(tile,truth,fixed);fc=assign(fullall,truth,[None,None]);tc=assign(tileall,truth,fc)
  if c['split']=='familiar':
   for a,b in zip(fixed,c['past_boxes']):assert (a is None)==(b is None) and (a is None or np.max(abs(np.array(a['box'])-b))<1e-5),(cid,a,b)
  for i,o in enumerate(objects):rows.append(dict(case=cid,video=v,split=c['split'],index=idx,slot=i,identity=o['id'],official_box=truth[i].tolist(),full=fixed[i],plus_tiles=combined[i],allclass_full=fc[i],allclass_tiles=tc[i],best_full_any=diag(fullall,truth[i]),best_full_car=diag(full,truth[i]),best_tile_car=diag(tile,truth[i]),best_tile_any=diag(tileall,truth[i]),new_from_tiles=fixed[i] is None and combined[i] is not None))
  remaining=len(tile)-sum(a is None and b is not None for a,b in zip(fixed,combined));dups=sum(any(iou(np.array(t['box']),np.array(f['box']))>=.7 for f in full) for t in tile);times.append(dict(case=cid,split=c['split'],full= tf,extra_tiles=tt,allclass_full=ta,allclass_tiles=tta,full_candidates=len(full),raw_tile_candidates=len(tile),additional_not_selected_for_two_references=remaining,duplicates_of_full=dups));raws[cid]=dict(full=full,tile=tile,fullall=fullall,tileall=tileall);print('DETECT_WINDOWS',cid,flush=True)
save('detection_result.json',dict(records=rows,frames=times,raw_boxes=raws,scope='Unmatched extras do not mean false detections of other objects. No pooling across detector classes in primary.'))
