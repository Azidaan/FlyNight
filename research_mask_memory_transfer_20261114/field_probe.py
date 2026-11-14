from common import *
import numpy as np,cv2,time
if sys.argv[1]=='prepare':
 save('field_protocol.json',dict(registered=now(),reason='Primary first-only memory did not meet recall on small daytarget. Test one fixed zoom and separate magnification from loss of context.',inputs={**read(OUT/'protocol.json')['inputs'],rel(Path(__file__)):sha(__file__),rel(OUT/'result.json'):sha(OUT/'result.json')},selection='All three originalvideos and allninequeries, no exclusion. Window from past initial target geometry only: fixed540x540 nativepixels centeredat initial cropcenter, shiftedjustenoughtofit image. No queryguide/label used.',arms=['ContextOnly: retain exact originalvideoheight/width and objectpixels; replace outside fixed540window by neutral ImageNet mean RGB [124,116,104].','Zoom: crop exactly that same540window; official SAM resize to1024 now magnifies object. Restore mask to originalframe with zerosoutside.','Compare with saved fullframe ordinary/first-only. Both new image treatments use losslessPNG roundtrip of decoded JPEG; model-side image tensors read by official SAM _load_img_as_tensor, replacing only its extension dispatcher to accept PNG. Pixel-equivalence and sourcehashes checked.'],scope='Post-primary explanatory factorial, not heldout validation. Zoom changes scale, square aspect and frame coordinates, so result attributed to input rendering as a whole. Exact currentlabels only used after inference. No thresholds changed.',deadline='2026-11-14T08:55:00+03:00'))
 journal('Зарегистрирована проверка поля наблюдения mask_memory_transfer_20261114: у дневной маленькой цели первый запрос не прошёл полноту. На тех же трёх видео сравниваются полный кадр, тот же масштаб с закрытым внешним контекстом и увеличение одного фиксированного окна540×540, выбранного только по начальному кадру. Источник: field_protocol.json. Эта последующая диагностика не заменяет отрицательный первичный итог и не использует проверочные рамки для наведения.')
 save('queue_field.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(OUT/n):sha(OUT/n) for n in ['field_probe.py','common.py','field_protocol.json']},jobs=[dict(entry=rel(OUT/'field_probe.py'),args=['run'],timeout=600)]));raise SystemExit
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2.sam2_video_predictor as predictor_module
from sam2.utils.misc import _load_img_as_tensor
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def png_loader(video_path,image_size,offload_video_to_cpu,async_loading_frames,compute_device):
 assert offload_video_to_cpu and not async_loading_frames;files=sorted(Path(video_path).glob('*.png'));assert len(files)==25;images=torch.zeros(25,3,image_size,image_size,dtype=torch.float32)
 for i,path in enumerate(files):images[i],h,w=_load_img_as_tensor(str(path),image_size)
 images-=torch.tensor([.485,.456,.406])[:,None,None];images/=torch.tensor([.229,.224,.225])[:,None,None];return images,h,w
predictor_module.load_video_frames=png_loader
p=read(OUT/'protocol.json');pp=read(OUT/'field_protocol.json');m=read(OUT/'frames.json')['records'];initials=np.load(OUT/'initial_masks.npz');preds={};scores={};windows={};times=[];start=time.perf_counter()
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/p['weights']),device='cuda',apply_postprocessing=False)
 for c in m:
  cid=c['id'];r=next(x for x in p['design']['records'] if x['id']==cid);a,b,e,f=r['crop'];l=max(0,min(740,(a+e)//2-270));t=max(0,min(180,(b+f)//2-270));windows[cid]=[l,t,l+540,t+540];orig=initials[cid];assert int(orig.sum())==int(orig[t:t+540,l:l+540].sum());queries=[r['frame']+x for x in [6,12,24]]
  for treatment in ['context','zoom']:
   folder=OUT/'field_frames'/cid/treatment;folder.mkdir(parents=True,exist_ok=False);initial=orig if treatment=='context' else orig[t:t+540,l:l+540]
   for idx,rec in enumerate(c['frames']):
    im=cv2.imread(str(ROOT/rec['path']));patch=im[t:t+540,l:l+540]
    if treatment=='context':render=np.empty_like(im);render[:]=[104,116,124];render[t:t+540,l:l+540]=patch
    else:render=patch
    dest=folder/f'{idx:05d}.png';assert cv2.imwrite(str(dest),render);assert np.array_equal(cv2.imread(str(dest)),render)
   for memory in ['ordinary','first_only']:
    t0=time.perf_counter();state=model.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);model.add_new_mask(state,0,1,initial)
    for idx,ids,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=24):
     assert ids==[1];frame=c['frames'][idx]['frame'];val=logits[0,0].detach().cpu().numpy();mask=val>0
     if frame in queries or idx==0:
      key=cid+'__'+treatment+'__'+memory+'__'+str(frame);scores[key]=val
      if treatment=='zoom':full=np.zeros(orig.shape,bool);full[t:t+540,l:l+540]=mask;mask=full
      preds[key]=mask
     if memory=='first_only':state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
    times.append(dict(case=cid,treatment=treatment,memory=memory,seconds=time.perf_counter()-t0));del state;torch.cuda.empty_cache();print('FIELD_ARM',cid,treatment,memory,flush=True)
with (OUT/'field_predictions.npz').open('xb') as f:np.savez_compressed(f,**preds)
with (OUT/'field_scores.npz').open('xb') as f:np.savez_compressed(f,**scores)
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
save('field_execution.json',dict(passed=True,windows=windows,times=times,source_unchanged=True,seconds=time.perf_counter()-start,scope='Explicit reference-only fixed fields; all predictions mappedto fullframe. Both treatments preserve decodedJPEG objectpixels before official modelresize.'))
