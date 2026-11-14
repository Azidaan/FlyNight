from common import *
import time,cv2,numpy as np
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');frames=read(OUT/'frames.json')['records']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
initial=np.load(ROOT/p['initial_mask'])['1159_target'].astype(bool);assert initial.shape==(720,1280)
scores={};predictions={};timings=[];t0=time.perf_counter()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 torch.cuda.reset_peak_memory_stats();model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 state=model.init_state(video_path=str(OUT/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);model.add_new_mask(state,frame_idx=0,obj_id=1,mask=initial);torch.cuda.synchronize();initialized=time.perf_counter();last=initialized
 for idx,ids,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=len(frames)-1):
  assert ids==[1],ids;torch.cuda.synchronize();current=time.perf_counter();a=logits[0,0].detach().cpu().numpy();frame=frames[idx]['frame'];predictions['sam__'+str(frame)]=a>0
  if frame in p['query_frames'] or idx==0:scores['sam__'+str(frame)]=a
  timings.append(dict(frame=frame,seconds=current-last));last=time.perf_counter();print('SAM_FRAME',frame,flush=True)
 gpu=dict(name=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());del state,model;torch.cuda.empty_cache()
sam_end=time.perf_counter();previous=None;flowmask=initial.astype('uint8');height,width=initial.shape;xx,yy=np.meshgrid(np.arange(width,dtype='float32'),np.arange(height,dtype='float32'));flowtimes=[]
for r in frames:
 im=cv2.imread(str(ROOT/r['path']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);t=time.perf_counter()
 if previous is not None:
  flow=cv2.calcOpticalFlowFarneback(gray,previous,None,.5,3,15,3,5,1.2,0);flowmask=cv2.remap(flowmask,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 predictions['flow__'+str(r['frame'])]=flowmask.astype(bool);predictions['static__'+str(r['frame'])]=initial;flowtimes.append(dict(frame=r['frame'],seconds=time.perf_counter()-t));previous=gray
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**predictions)
with (OUT/'scores.npz').open('xb') as f:np.savez_compressed(f,**scores)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('execution_result.json',dict(passed=True,source_unchanged=True,initialization_seconds=initialized-t0,sam_seconds=sam_end-t0,sam_per_frame=timings,flow_per_frame=flowtimes,gpu=gpu,threshold=0,total_seconds=time.perf_counter()-t0,scope='One bounded49-frame mask propagation; no project detector, identity assignment, FlyVis core or full tracker. Latencies are this clip, not a repeated benchmark.'))
print('PROPAGATION_FINISHED',flush=True)
