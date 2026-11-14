from official import *
from assignment import *
import torch,gzip,time
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
old=ROOT/'research_official_candidates_2179_20261114';other=ROOT/'research_depth_identity_competition_20261114';ar=np.load(other/'auto_candidates.npz');prev=np.load(ROOT/'research_auto_prior_masks_20261114/predictions.npz');candidates=np.load(old/'candidate_masks.npz');rt=read(ROOT/'research_assignment_weights_2179_20261112/runtime.json');record=[];observations={};prior={};sec=[];w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt');cfg='configs/sam2.1/sam2.1_hiera_t.yaml'
def bounds(m):
 y,x=np.where(m);return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float) if len(y) else None
def scorebox(b,boxes):
 return [iou(np.array(b),np.array(bb)) if b is not None else 0. for bb in boxes]
def solve(case,v,fr,ids,ref_boxes,prior_masks,items,obs_masks,protected,occupied_boxes):
 rids=[r['id'] for r in items];boxes=[r['box'] for r in items]
 # An unoccupied near-duplicate of an already occupied original remains unavailable.
 block={rid:own for rid,own in protected.items()}
 for r in items:
  if r['id'] not in block and any(iou(np.array(r['box']),np.array(b))>=.7 for b in occupied_boxes):block[r['id']]='reserved_duplicate'
 methods={'past_box':np.array([scorebox(b,boxes) for b in ref_boxes]),'predicted_box':np.array([scorebox(bounds(m),boxes) for m in prior_masks]),'mask':overlap_matrix(prior_masks,obs_masks)}
 actions={}
 for name,S in methods.items():
  a=edges(S,ids,rids,protected=block);actions[name]=dict(pairs=a.pairs,deferred=a.deferred,scores=a.scores)
 for i,m in enumerate(prior_masks):prior[f'{case}__{fr}__{i}']=m
 for r,m in zip(items,obs_masks):observations[f"{case}__{fr}__{r['id']}"]=m
 record.append(dict(case=case,video=v,frame=fr,owners=ids,items=items,protected=list(protected.items()),reserved=list(block.items()),actions=actions,past_boxes=ref_boxes))
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False)
 refs=[next(r for r in next(e for e in rt['events'] if e['frame']==1160)['rows'] if r['tid']==tid)['reference']['box'] for tid in (2179,2445)]
 state=vp.init_state(str(ROOT/'research_mask_propagation_20261114/frames'),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);vp.add_new_points_or_box(state,0,1,box=np.array(refs[1],'float32'));near={}
 for j,_,z in vp.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=12):
  if j in (6,12):near[1159+j]=(z[0,0]>0).cpu().numpy()
  state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
 del vp,state;torch.cuda.empty_cache()
 archive=read(old/'candidate_result.json')['records'];v=read(old/'protocol.json')['video'];hist=json.load(gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz'))
 for fr in [1165,1171]:
  pool=[r for r in archive if r['frame']==fr];rr=[r for r in pool if r['confidence']>.1];occupied={int(q['raw_id']):int(q['tid']) for q in hist['assignments'] if q['frame']==fr and q['emitted']}
  raw=np.load(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz');occupied_boxes=[raw['xyxy'][rid] for rid in occupied if raw['cls'][rid]==2]
  solve('old2179',v,fr,[2179,2445],refs,[prev[f'old2179__box__{fr}'],near[fr]],[dict(id=r['raw_id'],box=r['box'],confidence=r['confidence']) for r in rr],[candidates[f"{fr}__{r['raw_id']}"] for r in rr],occupied,occupied_boxes)
 imnet=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for c in read(other/'pairs.json')['cases']:
  cid=c['id'];v=c['video']
  for j in [1,2,4]:
   im=cv2.imread(str(ROOT/c['frames'][j]['image']));start=time.perf_counter();d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];boxes=d.boxes.xyxy.cpu().numpy();conf=d.boxes.conf.cpu().numpy();items=[dict(id=i,box=b.tolist(),confidence=float(conf[i])) for i,b in enumerate(boxes) if conf[i]>.1];m=[]
   if items:
    imnet.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    for item in items:m.append(imnet.predict(box=np.array(item['box'],'float32'),multimask_output=False)[0][0].astype(bool))
   torch.cuda.synchronize();sec.append(time.perf_counter()-start)
   solve(cid,v,j,[0,1],c['past_boxes'],[ar[f'{cid}__{i}__{j}'] for i in (0,1)],items,m,{},[])
  print('ASSOCIATION_CONSTRUCTED',cid,flush=True)
for n,a in [('observations.npz',observations),('prior.npz',prior)]:
 with (OUT/n).open('xb') as f:np.savez_compressed(f,**a)
save('decision.json',dict(records=record,observation_seconds=sec,unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
