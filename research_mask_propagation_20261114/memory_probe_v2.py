from common import *
import numpy as np,cv2,time
if sys.argv[1]=='prepare':
 save('memory_protocol.json',dict(registered=now(),reason='Primary SAM mask moves with foreground neighbor and then disappears while target remains visible. Intervene on whether self-generated nonconditioning memory persists.',inputs={**read(OUT/'protocol.json')['inputs'],rel(Path(__file__)):sha(__file__),rel(OUT/'result.json'):sha(OUT/'result.json'),rel(OUT/'predictions.npz'):sha(OUT/'predictions.npz')},arms=['Exact fresh-state baseline replay, same49frames and firstmanualmask','Same49frames, weights, mask, indices and processing, but clear non_cond_frame_outputs after each yielded frame. Keep first conditioning memory untouched. This removes previous predicted maskfeatures and objectpointers, not the currentimage or initialtarget.'],acceptance='A replay must reproduce primary masks. Improvement with only initialmemory demonstrates a causal contribution of self-generated temporal memory to these errors; failure to improve does not prove its innocence or all information absence. Keep original negativeoutcome.',posthoc_annotation='Additional negative core on1171near moving vehicle rearwindow: detail coordinates [(867,185),(1135,181),(1135,266),(864,267)]. Added after seeing failure to measure drift, NOT used to replace primarytest or fit. It fills a gap in the original sparse negativecoverage.',source='tools/sam2-runtime-20261027/sam2/modeling/sam2_base.py _prepare_memory_conditioned_features reads both predicted maskmem_features and obj_ptr from non_cond_frame_outputs.',deadline='2026-11-14T08:55:00+03:00'))
 journal('Зарегистрирована проверка памяти mask_propagation_20261114: основнойSAM2после хорошего1165теряет цель на более поздних запросах. Сравнить точный повтор с удалением только самостоятельно предсказанной памяти при сохранении первого ручного образца, текущих кадров и весов. Дополнительная отрицательная область на1171помечена как диагностическая после результата; первичная разметка и критерии остаются неизменными. Источник: memory_protocol.json.')
 save('queue_memory.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(OUT/n):sha(OUT/n) for n in ['memory_probe.py','common.py','memory_protocol.json']},jobs=[dict(entry=rel(OUT/'memory_probe.py'),args=['run'],timeout=600)]));raise SystemExit
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');pp=read(OUT/'memory_protocol_v2.json');frames=read(OUT/'frames.json')['records'];primary=np.load(OUT/'predictions.npz');initial=np.load(ROOT/p['initial_mask'])['1159_target'].astype(bool);predictions={};scores={};traces=[];start=time.perf_counter()
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
def fingerprint(value):
 h=hashlib.sha256()
 def update(v):
  if isinstance(v,torch.Tensor):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v):h.update(str(k).encode());update(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:update(x)
  else:h.update(repr(v).encode())
 update(value);return h.hexdigest()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 for variant in ['replay','initial_memory_only']:
  state=model.init_state(str(OUT/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);model.add_new_mask(state,0,1,initial);condhash=None
  for idx,ids,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=len(frames)-1):
   assert ids==[1];a=logits[0,0].detach().cpu().numpy();frame=frames[idx]['frame'];mask=a>0;predictions[variant+'__'+str(frame)]=mask
   if frame in p['query_frames']:scores[variant+'__'+str(frame)]=a
   if variant=='replay':assert np.array_equal(mask,primary['sam__'+str(frame)]),(variant,frame)
   output=state['output_dict_per_obj'][0];assert list(output['cond_frame_outputs'])==[0]
   if idx==0:condhash=fingerprint(output['cond_frame_outputs'])
   trace=dict(variant=variant,frame=frame,predicted_pixels=int(mask.sum()),nonconditioning_before_clear=list(output['non_cond_frame_outputs']))
   if variant=='initial_memory_only':output['non_cond_frame_outputs'].clear();assert not output['non_cond_frame_outputs']
   trace['nonconditioning_after_clear']=list(output['non_cond_frame_outputs']);traces.append(trace)
  assert fingerprint(output['cond_frame_outputs'])==condhash;del state;torch.cuda.empty_cache();print('MEMORY_VARIANT_FINISHED',variant,flush=True)
with (OUT/'memory_predictions.npz').open('xb') as f:np.savez_compressed(f,**predictions)
with (OUT/'memory_scores.npz').open('xb') as f:np.savez_compressed(f,**scores)
neg=cv2.fillPoly(np.zeros((720,1280),'uint8'),[np.rint(np.array([[867,185],[1135,181],[1135,266],[864,267]])/3+[900,200]).astype('int32')],1).astype(bool)
drift=[]
for variant,mask in [('primary',primary['sam__1171']),('initial_memory_only',predictions['initial_memory_only__1171'])]:drift.append(dict(variant=variant,diagnostic_negative_pixels=int(neg.sum()),predicted_on_diagnostic_neighbor=int((mask&neg).sum()),predicted_pixels=int(mask.sum()),fraction_of_prediction_on_diagnostic_neighbor=float((mask&neg).sum()/mask.sum()) if mask.any() else None))
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
save('memory_result.json',dict(passed=True,baseline_replay_exact=True,initial_conditioning_memory_unchanged=True,traces=traces,drift_diagnostic=drift,seconds=time.perf_counter()-start,source_unchanged=True));print('MEMORY_PROBE_FINISHED',flush=True)
