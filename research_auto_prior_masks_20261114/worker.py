from common import *
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
import cv2,numpy as np,time
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');old=ROOT/'research_mask_propagation_20261114';early=ROOT/'research_early_target_parts_20261114';multi=ROOT/'research_mask_memory_transfer_20261114';manual_multi=np.load(multi/'initial_masks.npz');expected_multi=np.load(multi/'predictions.npz');expected_old=np.load(early/'predictions.npz')
outputs={};prompts={};rois={};selections=[];traces=[];perf=[];start=time.monotonic()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 torch.cuda.reset_peak_memory_stats();teacher=YOLO(str(ROOT/p['segweights']))
 for c in p['cases']:
  for ref in c['ref']:
   key=c['id']+'__'+str(ref['frame']);im=cv2.imread(str(ROOT/ref['image']));result=teacher.predict(im,**p['settings'])[0];box=np.array(ref['box']);boxes=result.boxes.xyxy.cpu().numpy();masks=result.masks.data.cpu().numpy().astype(bool) if result.masks is not None else np.empty((0,*im.shape[:2]),bool)
   assert len(boxes)==len(masks);inter=np.maximum(np.minimum(boxes[:,2:],box[2:])-np.maximum(boxes[:,:2],box[:2]),0).prod(1);areas=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1);overlap=inter/np.maximum(areas+np.prod(box[2:]-box[:2])-inter,1e-12);choose=int(np.argmax(overlap)) if len(overlap) and overlap.max()>0 else None
   y=masks[choose] if choose is not None else np.zeros(im.shape[:2],bool);prompts[key+'__yolo']=y
   for i,a in enumerate(masks):rois[key+'__'+str(i)]=a
   if c['id']=='old2179':m=(np.load(early/'prior_parts.npz')['visible'] if ref['frame']==1141 else np.load(ROOT/read(old/'protocol.json')['initial_mask'])['1159_target'])
   else:m=manual_multi[c['id']]
   prompts[key+'__manual']=m.astype(bool);selections.append(dict(key=key,source_box=ref['box'],boxes=boxes.tolist(),scores=result.boxes.conf.cpu().numpy().tolist(),box_iou=overlap.tolist(),selected=choose,mask_pixels=int(y.sum()),manual_prompt_coverage=float((y&m.astype(bool)).sum()/m.sum())))
 del teacher;torch.cuda.empty_cache();sam=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 for c in p['cases']:
  cid=c['id'];first=c['frames'][0]['frame'];end=c['frames'][-1]['frame']
  for src in ['manual','yolo','box']:
   t=time.monotonic();state=sam.init_state(str(ROOT/c['folder']),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);idxs=[]
   for ref in c['ref']:
    idx=ref['frame']-first;key=cid+'__'+str(ref['frame'])
    if src=='box':_,_,v=sam.add_new_points_or_box(state,idx,1,box=np.array(ref['box'],dtype='float32'))
    else:
     mask=prompts[key+'__'+src]
     if not mask.any():continue
     _,_,v=sam.add_new_mask(state,idx,1,mask)
    prompts[key+'__'+src+'_decoded']=v[0,0].detach().cpu().numpy()>0;idxs.append(idx)
   if idxs:
    for idx,ids,logits in sam.propagate_in_video(state,start_frame_idx=idxs[-1],max_frame_num_to_track=end-first-idxs[-1]):
     assert ids==[1];fr=first+idx
     if fr in c['queries']:outputs[f'{cid}__{src}__{fr}']=(logits[0,0]>0).cpu().numpy()
     oc=state['output_dict_per_obj'][0];assert set(oc['cond_frame_outputs'])==set(idxs);oc['non_cond_frame_outputs'].clear()
   else:
    for fr in c['queries']:outputs[f'{cid}__{src}__{fr}']=np.zeros((720,1280),bool)
   if src=='manual':
    for fr in c['queries']:
     expected=expected_old['both__first_only__'+str(fr)] if cid=='old2179' else expected_multi[cid+'__first_only__'+str(fr)];assert np.array_equal(expected,outputs[f'{cid}__{src}__{fr}'])
   torch.cuda.synchronize();perf.append(dict(case=cid,source=src,seconds=time.monotonic()-t,reference_frames_used=[first+i for i in idxs],peak_allocated_bytes=torch.cuda.max_memory_allocated()));del state;torch.cuda.empty_cache();print('AUTO_PRIOR',cid,src,idxs,flush=True)
for file,arrays in [('predictions.npz',outputs),('prompts.npz',prompts),('all_teacher_masks.npz',rois)]:
 with (OUT/file).open('xb') as f:np.savez_compressed(f,**arrays)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('execution_result.json',dict(passed=True,replayed_manual_queries=14,selections=selections,timings=perf,seconds=time.monotonic()-start))
