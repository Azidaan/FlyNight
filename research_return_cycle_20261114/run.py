from official import *
from assignment import edges,overlap_matrix
import time,torch,gc
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');past=ROOT/'research_mask_ownership_adapter_20261114';holds=ROOT/'research_depth_identity_competition_20261114';om=np.load(past/'observations.npz');op=np.load(past/'prior.npz');ar=np.load(holds/'auto_candidates.npz');saved={(q['case'],q['frame']):q for q in read(past/'decision.json')['records']};pred={};returns={};records=[];status=[];zero=np.zeros((720,1280),bool)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def detect(im):
 d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];return [dict(id=i,box=b.tolist(),confidence=float(conf)) for i,(b,conf) in enumerate(zip(d.boxes.xyxy.cpu().numpy(),d.boxes.conf.cpu().numpy()))]
def propagate(folder,j,box_or_mask,start,end,back):
 st=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 if box_or_mask.ndim==1:vp.add_new_points_or_box(st,j,1,box=box_or_mask.astype('float32'))
 else:vp.add_new_mask(st,j,1,box_or_mask.astype(bool))
 out={}
 for k,_,z in vp.propagate_in_video(st,start_frame_idx=start,max_frame_num_to_track=abs(end-start),reverse=back):
  if k in [0,1,2,4] or k==end:out[k]=(z[0,0]>0).cpu().numpy()
  st['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
 return out
def sub(cid,j,rids,ms,pa,anchors,forward,folder,offset,record):
 proposed=forward;ret_by={};start=time.perf_counter()
 for i,rid in forward:
  m=ms[rids.index(rid)]
  b=propagate(folder,offset,m,offset,0,True)[0];ret_by[rid]=b;returns[f'{cid}__{j}__{rid}']=b
 rt=overlap_matrix(anchors,list(ret_by.values()));good=list(edges(rt,[0,1],list(ret_by)).pairs)
 safe=[[i,rid] for i,rid in proposed if (i,rid) in good];record.update(forward=forward,confirmed=safe,backward=good,return_ids=list(ret_by),return_scores=rt.tolist(),cycle_seconds=time.perf_counter()-start)
 for i,b in enumerate(anchors):pred[f'{cid}__past__{i}']=b
 for i,rid in forward:pred[f'{cid}__{j}__obs__{rid}']=ms[rids.index(rid)]
 records.append(record)
cfg='configs/sam2.1/sam2.1_hiera_t.yaml';w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt')
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);imp=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for c in read(OUT/'cases.json')['cases']:
  cid=c['id'];folder=(ROOT/c['frames'][0]['image']).parent
  if c['split']=='diagnostic':
   boxes=c['past_boxes'];anchors=[ar[f'{cid}__{i}__0'] for i in [0,1]]
  else:
   ds=detect(cv2.imread(str(ROOT/c['frames'][0]['image'])));jacc=np.array([[iou(np.array(b),np.array(t['box'])) for t in ds] for b in c['past_truth_boxes']]);cost=np.where(jacc>=.5,1+jacc,0);ii,jj=linear_sum_assignment(-np.c_[cost,np.zeros((2,2))]);boxes=[None,None]
   for i,j in zip(ii,jj):
    if j<len(ds) and cost[i,j]>0:boxes[i]=ds[j]['box']
   m=[propagate(folder,0,np.array(b),0,4,False) if b is not None else {k:zero for k in [0,1,2,4]} for b in boxes];anchors=[a[0] for a in m]
  for j in [1,2,4]:
   if c['split']=='diagnostic':
    q=saved[cid,j];items=q['items'];ms=[om[f"{cid}__{j}__{r['id']}"] for r in items];pa=[op[f'{cid}__{j}__{i}'] for i in [0,1]];expected=q['actions']['mask']['pairs']
   else:
    im=cv2.imread(str(ROOT/c['frames'][j]['image']));items=[x for x in detect(im) if x['confidence']>.1];ms=[];pa=[x[j] for x in m];expected=None
    if items:
     imp.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
     for x in items:ms.append(imp.predict(box=np.array(x['box'],'float32'),multimask_output=False)[0][0].astype(bool))
   ids=[x['id'] for x in items];S=overlap_matrix(pa,ms);fwd=[list(x) for x in edges(S,[0,1],ids).pairs]
   if expected is not None:assert fwd==expected
   sub(cid,j,ids,ms,pa,anchors,fwd,folder,j,dict(case=cid,video=c['video'],frame=j,index=c['index']+j,split=c['split'],ids=c['ids'],past_boxes=boxes))
  print('REVERSE_CHECK',cid,flush=True)
 # Old requested 2179 comparison, independent one-step readback of the same accepted proposal.
 oldfolder=ROOT/'research_mask_propagation_20261114/frames';oo=np.load(ROOT/'research_official_candidates_2179_20261114/candidate_masks.npz');oldrec=read(ROOT/'research_official_candidates_2179_20261114/candidate_result.json')['records'];anchors=[]
 for tid in (2179,2445):
  x=next(x for x in oldrec if x['frame']==1159 and x['assigned_tid']==tid);anchors.append(oo['1159__'+str(x['raw_id'])])
 for fr in [1165,1171]:
  q=saved['old2179',fr];ids=[b['id'] for b in q['items']];m=[om[f'old2179__{fr}__{i}'] for i in ids];fwd=[[0 if tid==2179 else 1,rid] for tid,rid in q['actions']['mask']['pairs']];sub('old2179',fr,ids,m,[],anchors,fwd,oldfolder,fr-1159,dict(case='old2179',video=q['video'],frame=fr,index=None,split='old',ids=[2179,2445],past_boxes=q['past_boxes']))
for n,a in [('outputs.npz',pred),('returns.npz',returns)]:
 with (OUT/n).open('xb') as f:np.savez_compressed(f,**a)
save('decisions.json',dict(records=records,source_hashes_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
