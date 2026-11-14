from common import *
import numpy as np,time
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');early=np.load(OUT/'prior_parts.npz');previous=ROOT/'research_mask_propagation_20261114';late=np.load(ROOT/read(previous/'protocol.json')['initial_mask'])['1159_target'].astype(bool);replay=np.load(previous/'predictions.npz');replay_frozen=np.load(previous/'memory_predictions.npz');pred={};scores={};log=[];start=time.perf_counter()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 arms=[(source,mode) for source in ['late','early','both'] for mode in ['ordinary','first_only']]+[(x,'first_only') for x in ['front_equal','rear_equal']]
 for source,mode in arms:
  t=time.perf_counter();initial=1159 if source=='late' else 1141;folder=previous/'frames' if source=='late' else OUT/'frames';state=model.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
  if source=='late':model.add_new_mask(state,0,1,late)
  else:model.add_new_mask(state,0,1,early['visible'] if source in ['early','both'] else early[source]);
  if source=='both':model.add_new_mask(state,1159-1141,1,late)
  # All measured queries are after the last prompt. Start at latest available
  # prompt for both, so no future conditioning enters an earlier prediction.
  start_idx=18 if source=='both' else 0;last=1207-initial
  for idx,ids,logits in model.propagate_in_video(state,start_frame_idx=start_idx,max_frame_num_to_track=last-start_idx):
   assert ids==[1];fr=initial+idx;value=logits[0,0].detach().cpu().numpy();mask=value>0
   if fr in p['queries']:
    key=source+'__'+mode+'__'+str(fr);pred[key]=mask;scores[key]=value
    if source=='late':assert np.array_equal(mask,replay['sam__'+str(fr)] if mode=='ordinary' else replay_frozen['initial_memory_only__'+str(fr)]),(source,mode,fr)
   state_output=state['output_dict_per_obj'][0];assert set(state_output['cond_frame_outputs'])==({0,18} if source=='both' else {0})
   if mode=='first_only':state_output['non_cond_frame_outputs'].clear()
  torch.cuda.synchronize();log.append(dict(source=source,mode=mode,seconds=time.perf_counter()-t,initial_frame=initial,start_frame=initial+start_idx,known_prompts=[1141,1159] if source=='both' else [initial]));del state;torch.cuda.empty_cache();print('EARLY_ARM_FINISHED',source,mode,flush=True)
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**pred)
with (OUT/'scores.npz').open('xb') as f:np.savez_compressed(f,**scores)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('execution_result.json',dict(passed=True,late_baseline_exact_on_5_queries_both_modes=True,timings=log,source_unchanged=True,seconds=time.perf_counter()-start));print('EARLY_PARTS_FINISHED')
