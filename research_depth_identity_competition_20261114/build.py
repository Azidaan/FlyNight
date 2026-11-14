from official import *
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from tracking.gpu_lock import GpuLock
from scipy.optimize import linear_sum_assignment
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=read(OUT/'protocol.json');lab=zipfile.ZipFile(ROOT/p['labelzip']);cache={};cases=[];skips=[];auto={};oracle={};record=[];blank=lambda:np.zeros((720,1280),bool)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'));sam=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
 for c in p['cases']:
  cid,v,idx=c['id'],c['video'],c['index']
  if v not in cache:cache[v]=labels(lab,v)
  r=cache[v][idx];target=next(o for o in r['labels'] if o['id']==c['target']);others=[o for o in r['labels'] if eligible(o) and o['id']!=c['target'] and M.area(o['rle'])>0]
  if not others:skips.append(dict(case=cid,reason='no second eligible car'));continue
  rival=sorted(others,key=lambda o:(-iou(bbox(target),bbox(o)),-int(M.area(o['rle'])),o['id']))[0];pair=[target,rival];ids=[o['id'] for o in pair];im=cv2.imread(str(ROOT/c['frames'][0]['image']));d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];boxes=d.boxes.xyxy.cpu().numpy();J=np.array([[iou(bbox(o),bb) for bb in boxes] for o in pair]);cost=np.where(J>=.5,1+J,0);ri,ci=linear_sum_assignment(-np.c_[cost,np.zeros((2,2))]);refs=[None,None]
  for i,j in zip(ri,ci):
   if j<len(boxes) and J[i,j]>=.5:refs[i]=boxes[j].tolist()
  for i,obj in enumerate(pair):
   if refs[i] is not None:
    state=sam.init_state(str(ROOT/c['folder']),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);sam.add_new_points_or_box(state,0,1,box=np.array(refs[i],'float32'))
    for j,_,val in sam.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=4):
     if j in [0,1,2,4]:auto[f'{cid}__{i}__{j}']=(val[0,0]>0).cpu().numpy()
     state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
    del state
   else:
    for j in [0,1,2,4]:auto[f'{cid}__{i}__{j}']=blank()
   for j in [0,1,2,4]:
    rr=cache[v][idx+j];t,_,_=masks(lab,v,rr,obj['id']);oracle[f'{cid}__{i}__{j}']=t
  cases.append(dict(id=cid,video=v,index=idx,ids=ids,past_boxes=refs,reference_box_iou=iou(bbox(target),bbox(rival)),frames=c['frames']));print('PAIR_READY',cid,flush=True)
for name,data in [('auto_candidates.npz',auto),('oracle_candidates.npz',oracle)]:
 with (OUT/name).open('xb') as f:np.savez_compressed(f,**data)
save('pairs.json',dict(cases=cases,excluded=skips,auto_sha256=sha(OUT/'auto_candidates.npz'),oracle_sha256=sha(OUT/'oracle_candidates.npz'),selection='Only past metadata determines chosen car pair. Separate oracle arrays are privileged; not used by automatic mask generation.'))
