from official import *
from assignment import edges,overlap_matrix
import time,torch
from scipy.optimize import linear_sum_assignment
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol_v2.json');z=zipfile.ZipFile(ROOT/p['labelzip']);cases=read(OUT/'cases.json')['cases'];rows=[];traces=[];verified={};zero=np.zeros((720,1280),bool)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for c in cases:
 for f in c['frames']:assert sha(ROOT/f['image'])==f['sha256']
def detect(im):
 d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];return [(i,b,float(s)) for i,(b,s) in enumerate(zip(d.boxes.xyxy.cpu().numpy(),d.boxes.conf.cpu().numpy()))]
def bounds(m):
 y,x=np.where(m);return np.array([x.min(),y.min(),x.max()+1,y.max()+1]) if len(x) else None
def ious(past,boxes):return np.array([[iou(bb,b) if bb is not None else 0. for b in boxes] for bb in past])
cfg='configs/sam2.1/sam2.1_hiera_t.yaml';w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt')
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);image_model=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for c in cases:
  cid=c['id'];v=c['video'];db=labels(z,v);fs=c['frames'];folder=(ROOT/fs[0]['image']).parent;N=len(fs)-1;ds=detect(cv2.imread(str(ROOT/fs[0]['image'])));dsboxes=[b for _,b,_ in ds];M0=ious([np.array(b) for b in c['past_boxes']],dsboxes);val=np.where(M0>=.5,1+M0,0);ii,jj=linear_sum_assignment(-np.c_[val,np.zeros((2,2))]);boxes=[None,None]
  for i,j in zip(ii,jj):
   if j<len(ds) and val[i,j]>0:boxes[i]=dsboxes[j]
  prior=[]
  for bb in boxes:
   if bb is None:prior.append({i:zero for i in range(N+1)});continue
   st=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);vp.add_new_points_or_box(st,0,1,box=bb);m0={}
   for k,_,l in vp.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=N):m0[k]=(l[0,0]>0).cpu().numpy();st['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
   prior.append(m0)
  for k in range(1,N+1):
   im=cv2.imread(str(ROOT/fs[k]['image']));pool=[x for x in detect(im) if x[2]>.1];rid=[x[0] for x in pool];bs=[x[1] for x in pool];pr=[m[k] for m in prior];masks_now=[]
   if pool:
    image_model.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    for _,b,_ in pool:masks_now.append(image_model.predict(box=b,multimask_output=False)[0][0].astype(bool))
   r=db[c['prior']+k];g=[masks(z,v,r,o) for o in [c['target'],c['rival']]];oracle=overlap_matrix([x[0] for x in g],[x&g[0][1] for x in masks_now]);method={'past_box':ious(boxes,bs),'predicted_box':ious([bounds(x) for x in pr],bs),'mask':overlap_matrix(pr,masks_now)};actions={}
   for name,s in method.items():
    result=edges(s,[0,1],rid);act=dict(result.pairs);actions[name]=list(result.pairs)
    for i,identity in enumerate([c['target'],c['rival']]):
     valid_obj=next((o for o in r['labels'] if o['id']==identity),None);ignored=valid_obj is not None and any((valid_obj.get('attributes') or {}).get(key,False) for key in ['ignore','ignored','crowd','iscrowd']);ind=rid.index(act[i]) if i in act else None;score=oracle[i,ind] if ind is not None else None;status='unknown_ignore' if ignored else 'wait' if ind is None else 'correct' if score>.5 else 'wrong';rows.append(dict(case=cid,video=v,frame=k,first_return=k==N,slot=i,id=identity,method=name,action=act.get(i),status=status,visible=bool(g[i][0].any()),ignored=ignored,target_iou=float(score) if score is not None else None,prior_available=boxes[i] is not None,available_current=bool(max(oracle[i],default=0)>.5),phase='return' if k==N else 'gap'))
     if name=='mask' and i==0 and (ind is not None or k==N):
      verified[f'{cid}__{k}__chosen']=masks_now[ind] if ind is not None else zero
   traces.append(dict(case=cid,frame=k,rids=rid,actions=actions,scores={n:s.tolist() for n,s in method.items()},overlap_with_truth=oracle.tolist()))
  print('REAL_GAP',cid,N,flush=True)
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**verified)
save('actions.json',dict(records=rows,traces=traces))
