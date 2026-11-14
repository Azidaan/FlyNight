from common import *
import cv2,numpy as np,time
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');manifest=read(OUT/'frames.json');initials=np.load(OUT/'initial_masks.npz');outputs={};scores={};timings=[];traces=[];start=time.perf_counter()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 torch.cuda.reset_peak_memory_stats();model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 for c in manifest['records']:
  cid=c['id'];initial=initials[cid];queries=[c['frames'][0]['frame']+off for off in [6,12,24]]
  for variant in ['ordinary','first_only']:
   t0=time.perf_counter();state=model.init_state(str(OUT/'frames'/cid),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);model.add_new_mask(state,0,1,initial)
   for idx,ids,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=24):
    assert ids==[1];frame=c['frames'][idx]['frame'];a=logits[0,0].detach().cpu().numpy();mask=a>0;key=cid+'__'+variant+'__'+str(frame);outputs[key]=mask
    if idx==0 or frame in queries:scores[key]=a
    output=state['output_dict_per_obj'][0];assert list(output['cond_frame_outputs'])==[0]
    if variant=='first_only':output['non_cond_frame_outputs'].clear();assert not output['non_cond_frame_outputs']
    if idx==0:
     intersection=int((mask&initial).sum());union=int((mask|initial).sum());traces.append(dict(case=cid,variant=variant,initial_pixels=int(initial.sum()),output_pixels=int(mask.sum()),initial_iou=intersection/union if union else None))
     if variant=='first_only':assert np.array_equal(mask,outputs[cid+'__ordinary__'+str(frame)])
   torch.cuda.synchronize();timings.append(dict(case=cid,variant=variant,seconds=time.perf_counter()-t0));del state;torch.cuda.empty_cache();print('VIDEO_ARM_FINISHED',cid,variant,flush=True)
 gpu=dict(name=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
for c in manifest['records']:
 cid=c['id'];initial=initials[cid];flowmask=initial.astype('uint8');previous=None;yy,xx=np.indices(initial.shape,dtype='float32')
 for r in c['frames']:
  image=cv2.imread(str(ROOT/r['path']));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  if previous is not None:
   flow=cv2.calcOpticalFlowFarneback(gray,previous,None,.5,3,15,3,5,1.2,0);flowmask=cv2.remap(flowmask,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  outputs[cid+'__flow__'+str(r['frame'])]=flowmask.astype(bool);outputs[cid+'__static__'+str(r['frame'])]=initial;previous=gray
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**outputs)
with (OUT/'scores.npz').open('xb') as f:np.savez_compressed(f,**scores)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('execution_result.json',dict(passed=True,timings=timings,initialization_checks=traces,gpu=gpu,source_unchanged=True,total_seconds=time.perf_counter()-start));print('MEMORY_TRANSFER_FINISHED',flush=True)
