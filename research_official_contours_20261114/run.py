from official import *
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch,time
from tracking.gpu_lock import GpuLock
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cv2.setNumThreads(2)
p=read(OUT/'protocol.json');cases=read(OUT/'cases.json');ini=np.load(OUT/'initial_official.npz');pred={};initial={};source={};missing=[];perf=[];start=time.perf_counter()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 dmodel=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'));yseg=YOLO(str(ROOT/'weights/yolo26seg_official_20261114/yolo26n-seg.pt'));base=read(ROOT/'research_auto_prior_masks_20261114/protocol.json')['settings']
 for c in cases['cases']:
  key=c['id'];g=np.array(c['past_box']);im=cv2.imread(str(ROOT/c['frames'][0]['image']));d=dmodel.predict(im,**dict(base,conf=.01))[0];bs=d.boxes.xyxy.cpu().numpy();scores=[iou(x,g) for x in bs];i=int(np.argmax(scores)) if scores and max(scores)>=.5 else None;bb=bs[i] if i is not None else None;seg=yseg.predict(im,**base)[0];ss=seg.boxes.xyxy.cpu().numpy();pairs=[iou(x,bb) for x in ss] if bb is not None else [];j=int(np.argmax(pairs)) if pairs and max(pairs)>0 else None
  initial[key+'__yolo']=seg.masks.data[j].cpu().numpy().astype(bool) if j is not None else np.zeros((720,1280),bool);source[key]=dict(detection_box=bb.tolist() if bb is not None else None,iou=max(scores,default=0.),det_idx=i,det_conf=float(d.boxes.conf[i]) if i is not None else None,seg_idx=j,seg_iou=max(pairs,default=0.));print('REFERENCE',key,i,j,flush=True)
 del dmodel,yseg;torch.cuda.empty_cache();sam=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
 for c in cases['cases']:
  ts=time.perf_counter();cid=c['id'];ref=source[cid]
  for mode in ['box_first','box_ordinary','yolo_first','manual_full_first']:
   bb=ref['detection_box'];empty=(mode.startswith('box') and bb is None) or (mode=='yolo_first' and not initial[cid+'__yolo'].any())
   if empty:
    for j in [0,1,2,4]:pred[f'{cid}__{mode}__{j}']=np.zeros((720,1280),bool)
    continue
   state=sam.init_state(str(ROOT/c['folder']),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
   if mode.startswith('box'):sam.add_new_points_or_box(state,0,1,box=np.array(bb,'float32'))
   else:sam.add_new_mask(state,0,1,ini[cid] if mode=='manual_full_first' else initial[cid+'__yolo'])
   for j,ids,val in sam.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=4):
    if j in [0,1,2,4]:pred[f'{cid}__{mode}__{j}']=(val[0,0]>0).cpu().numpy()
    if mode!='box_ordinary':state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
   del state;torch.cuda.empty_cache()
  first=pred[f'{cid}__box_first__0'];assert np.array_equal(first,pred[f'{cid}__box_ordinary__0']);cur=first.astype('uint8');last=None;yy,xx=np.indices(first.shape,'float32')
  for j,f in enumerate(c['frames']):
   assert sha(ROOT/f['image'])==f['sha256'];img=cv2.imread(str(ROOT/f['image']),cv2.IMREAD_GRAYSCALE)
   if last is not None:
    flow=cv2.calcOpticalFlowFarneback(img,last,None,.5,3,15,3,5,1.2,0);cur=cv2.remap(cur,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
   if j in [0,1,2,4]:pred[f'{cid}__flow_same_box_initial__{j}']=cur.astype(bool);pred[f'{cid}__static_same_box_initial__{j}']=first
   last=img
  perf.append(dict(case=cid,seconds=time.perf_counter()-ts));print('CASE_FINISHED',cid,flush=True)
 for file,data in [('predictions.npz',pred),('initial_teacher.npz',initial)]:
  with (OUT/file).open('xb') as f:np.savez_compressed(f,**data)
 save('execution_result.json',dict(passed=True,sources=source,timings=perf,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start,full_tracker_calls=0))
