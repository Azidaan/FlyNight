from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_VERBOSE']='false'
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import gc,time,cv2,numpy as np,torch,psutil
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from torchvision.ops import roi_align
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);cv2.setNumThreads(2)
stage=sys.argv[1];manifest=read(OUT/('case2179.json' if stage=='2179' else 'holdout/manifest.json'));p=read(OUT/'protocol.json')
dest=OUT/('stage_'+stage);dest.mkdir(exist_ok=False)
def put(name,data):
 with (dest/name).open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
put('execution_protocol.json',dict(started=now(),source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),manifest_sha256=sha(OUT/('case2179.json' if stage=='2179' else 'holdout/manifest.json')),inputs={r['image']:sha(ROOT/r['image']) for e in manifest['events'] for r in e['roles']}))
status=dict(started=now(),phase='initializing',stage=stage)
def progress(**kw):
 status.update(kw,updated=now());(dest/'status.json').write_text(json.dumps(status,ensure_ascii=False),encoding='utf8')
def iou_matrix(rois,boxes):
 a=np.asarray(rois,float).reshape(-1,4);b=np.asarray(boxes,float).reshape(-1,4)
 left=np.maximum(a[:,None,:2],b[None,:,:2]);right=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(0,right-left).prod(2)
 areas=lambda x:np.maximum(0,x[:,2:]-x[:,:2]).prod(1)
 return inter/np.maximum(areas(a)[:,None]+areas(b)[None,:]-inter,1e-12)
def assign(rois,boxes):
 matrix=iou_matrix(rois,boxes);n=len(rois);selected=[None]*n
 rows,cols=linear_sum_assignment(-np.concatenate([matrix,np.zeros((n,n))],axis=1))
 for row,col in zip(rows,cols):
  if col<len(boxes) and matrix[row,col]>0:selected[row]=int(col)
 return selected,matrix
def tensor_hash(model):
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
features={};records=[];masks_saved={};seg_records=[];timing=[];predictions={}
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 yolo=YOLO(str(ROOT/read(OUT/'weights_receipt.json')['path']));masa=build('cuda');assert not masa.training
 def segment(img):
  result=yolo.predict(source=img,**p['settings'])[0]
  boxes=result.boxes.xyxy.cpu().numpy().copy();conf=result.boxes.conf.cpu().numpy().copy();classes=result.boxes.cls.cpu().numpy().copy()
  masks=np.zeros((0,*img.shape[:2]),bool) if result.masks is None else result.masks.data.cpu().numpy().astype(bool)
  assert masks.shape==(len(boxes),*img.shape[:2])
  return dict(boxes=boxes,masks=masks,conf=conf,classes=classes)
 def descriptor(img,r,level):
  x,scale=preprocess(img,'cuda');ff=masa.features(x)
  boxes=torch.tensor(r.get('batch_boxes',[r['box']]),dtype=torch.float32,device='cuda');idx=r.get('batch_index',0)
  if level==-1:raw=masa.embed_boxes(ff,boxes,scale)
  else:
   roi=torch.cat([torch.zeros((len(boxes),1),device='cuda'),boxes*scale],1)
   pooled=roi_align(ff[level],roi,7,spatial_scale=1.0/(4*2**level),sampling_ratio=0,aligned=True);raw=masa.embed(pooled)
  return torch.nn.functional.normalize(raw,dim=1)[idx].cpu().numpy().copy()
 def changed(img,keep,fill):
  value=img.copy();value[~keep]=[104,116,124] if fill=='imagenet_mean' else [0,0,0]
  assert np.array_equal(value[keep],img[keep]);return value
 # Predictor initialization/fusion is completed before the frozen-parameter check.
 first=manifest['events'][0]['roles'][0];segment(cv2.imread(str(ROOT/first['image'])))
 yolo_core=yolo.predictor.model.model;assert not yolo_core.training
 model_hashes=dict(masa=tensor_hash(masa),seg=tensor_hash(yolo_core))
 put('device.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,end2end=bool(yolo.predictor.model.end2end),model_dtype=str(next(yolo_core.parameters()).dtype),model_parameter_count=sum(v.numel() for v in yolo_core.parameters()),after_fusion_hashes=model_hashes,settings=p['settings']))
 for event in manifest['events']:
  eid=event['id'];manual=np.load(ROOT/event['masks'],allow_pickle=False);roles=event['roles'];grouped={}
  for r in roles:grouped.setdefault(r['image'],[]).append(r)
  for path,rr in grouped.items():
   img=cv2.imread(str(ROOT/path));assert img is not None;pred=segment(img);selected,matrix=assign([r['box'] for r in rr],pred['boxes'])
   gkey=f"{eid}_{rr[0]['frame']}";masks_saved[gkey+'_all_auto']=pred['masks']
   seg_records.append(dict(event=eid,image=path,frame=rr[0]['frame'],mask_bank_key=gkey+'_all_auto',boxes=pred['boxes'].tolist(),scores=pred['conf'].tolist(),classes=pred['classes'].tolist(),roles=[r['role'] for r in rr],iou_matrix=matrix.tolist(),selected=selected,independent_top1=[int(np.argmax(x)) if len(x) and max(x)>0 else None for x in matrix]))
   for r,index in zip(rr,selected):
    key=eid+'_'+r['role'];auto=np.zeros(img.shape[:2],bool) if index is None else pred['masks'][index]
    predictions[key]=(auto,index,rr);masks_saved[key+'_automatic']=auto;masks_saved[key+'_manual']=manual[r['manual_key']].astype(bool);masks_saved[key+'_other']=manual[r['other_manual_key']].astype(bool)
  for r in roles:
   key=eid+'_'+r['role'];img=cv2.imread(str(ROOT/r['image']));original_hash=hashlib.sha256(img.tobytes()).hexdigest();auto,index,_=predictions[key]
   rect=np.zeros(img.shape[:2],bool);x1,y1,x2,y2=r['box'];h,w=rect.shape;rect[max(0,int(np.floor(y1))):min(h,int(np.ceil(y2))),max(0,int(np.floor(x1))):min(w,int(np.ceil(x2)))]=True
   keeps=dict(full=np.ones(img.shape[:2],bool),rectangle=rect,manual=masks_saved[key+'_manual'],automatic=auto)
   for arm,keep in keeps.items():
    for fill in p['fills']:
     modified=changed(img,keep,fill)
     for level in p['levels']:
      outkey=f'{key}_{arm}_{fill}_{level}';v=descriptor(modified,r,level);assert np.isfinite(v).all() and v.shape==(256,)
      features[outkey]=v;records.append(dict(key=outkey,event=eid,role=r['role'],arm=arm,fill=fill,level=level,selected_mask=index if arm=='automatic' else None,missing=bool(index is None) if arm=='automatic' else False,retained_pixels=int(keep.sum())))
   assert hashlib.sha256(img.tobytes()).hexdigest()==original_hash
   progress(phase='descriptors',event=eid,completed=len(records))
  # Benchmark current query only: references are cached, as in association.
  query=next(r for r in roles if r['role']=='query');qkey=eid+'_query';qimg=cv2.imread(str(ROOT/query['image']));rr=predictions[qkey][2];qi=next(i for i,v in enumerate(rr) if v['role']=='query')
  def pipeline(arm):
   if arm=='full':inp=qimg
   elif arm=='manual':inp=changed(qimg,masks_saved[qkey+'_manual'],'imagenet_mean')
   else:
    seg=segment(qimg);sel,_=assign([r['box'] for r in rr],seg['boxes']);idx=sel[qi];keep=np.zeros(qimg.shape[:2],bool) if idx is None else seg['masks'][idx];inp=changed(qimg,keep,'imagenet_mean')
   return descriptor(inp,query,0)
  for _ in range(p['warmup']):
   for arm in ['full','manual','automatic']:pipeline(arm)
  torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
  for rep in range(p['repeats']):
   for arm in ['full','manual','automatic']:
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();pipeline(arm);torch.cuda.synchronize();elapsed=time.perf_counter()-start
    mem=psutil.Process().memory_info();timing.append(dict(event=eid,arm=arm,repeat=rep,seconds=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),process_rss_bytes=mem.rss,process_peak_wset_bytes=getattr(mem,'peak_wset',None)))
   progress(phase='benchmark',event=eid,completed=rep+1,total=p['repeats'])
 assert tensor_hash(masa)==model_hashes['masa'];assert tensor_hash(yolo_core)==model_hashes['seg']
with (dest/'vectors.npz').open('xb') as f:np.savez_compressed(f,**features)
with (dest/'masks.npz').open('xb') as f:np.savez_compressed(f,**masks_saved)
put('results.json',dict(passed=True,stage=stage,records=records,segments=seg_records,timing=timing,model_unchanged=True,vectors_sha256=sha(dest/'vectors.npz'),masks_sha256=sha(dest/'masks.npz'),benchmark_scope='Current query only, cached reference and image I/O; includes preprocessing, mask decoding/CPU transfer/matching, masking and frozen MASA level0. Manual-mask creation time excluded. Both networks resident for all arms; allocator reserved memory can retain previous arms. Not full-tracker speed.'))
progress(phase='completed',completed=len(records));print('STAGE_COMPLETE',stage,len(records),flush=True)
