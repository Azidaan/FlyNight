from official import *
from assignment import overlap_matrix,edges
import torch
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');parent=ROOT/'research_return_cycle_20261114';cases=read(parent/'cases.json')['cases'];source={(q['case'],q['frame']):q for q in read(parent/'decisions.json')['records']};truths=read(parent/'result.json')['records'];prior=np.load(parent/'outputs.npz');lab=zipfile.ZipFile(ROOT/p['labelzip']);cfg='configs/sam2.1/sam2.1_hiera_t.yaml';w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt');records=[];tables=[];samples={};neighbors=[];zero=np.zeros((720,1280),np.float32)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'));imnet=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));assert not vp.non_overlap_masks
 for c in cases:
  cid=c['id'];v=c['video'];cache=labels(lab,v);box=source[cid,1]['past_boxes'];folder=(ROOT/c['frames'][0]['image']).parent;pred={}
  for memory in ['fixed','ordinary']:
   # Same one-object state and embeddings as baseline; retain real-valued scores for inter-object competition at readout only.
   for i,b in enumerate(box):
    if b is None:
     for j in [0,1,2,4]:pred[memory,i,j]=np.full_like(zero,-1024)
     continue
    st=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);vp.add_new_points_or_box(st,0,1,box=np.array(b,'float32'))
    for j,_,val in vp.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=4):
     if j in [0,1,2,4]:pred[memory,i,j]=val[0,0].cpu().numpy().astype('float32')
     if memory=='fixed':st['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
    del st
  for i in [0,1]:assert np.array_equal(pred['fixed',i,0]>0,prior[f'{cid}__past__{i}'])
  for j in [1,2,4]:
   im=cv2.imread(str(ROOT/c['frames'][j]['image']));d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];pool=[(ii,bb) for ii,(bb,cc) in enumerate(zip(d.boxes.xyxy.cpu().numpy(),d.boxes.conf.cpu().numpy())) if cc>.1];ids=[ii for ii,bb in pool];ms=[];imnet.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
   for ii,bb in pool:ms.append(imnet.predict(box=bb,multimask_output=False)[0][0].astype(bool))
   g=[masks(lab,v,cache[c['index']+j],identity) for identity in c['ids']];O=overlap_matrix([t for t,_,_ in g],[m&g[0][1] for m in ms]);source_row=source[cid,j]
   for memory in ['fixed','ordinary']:
    logits=np.stack([pred[memory,i,j] for i in [0,1]]);out=vp._apply_non_overlapping_constraints(torch.from_numpy(logits[:,None]))[:,0].numpy()>0;assert not np.any(out[0]&out[1]);manual=(logits>0)&(np.arange(2)[:,None,None]==np.argmax(logits,axis=0));assert np.array_equal(out,manual)
    for quality,P in [('separate',logits>0),('exclusive',out)]:
     name=memory+'_'+quality;s=overlap_matrix(P,ms);pairs=dict(edges(s,[0,1],ids).pairs)
     if name=='fixed_separate':assert pairs==dict(source_row['forward'])
     for i in [0,1]:
      rid=pairs.get(i);current=g[i][0];q=O[i,ids.index(rid)] if rid is not None else None;state='deferred' if rid is None else 'correct' if q>.5 else 'wrong';records.append(dict(case=cid,video=v,split=c['split'],offset=j,slot=i,arm=name,action=rid,status=state,target_iou=float(q) if q is not None else None,prior_iou=measure(P[i],*g[i])['iou'],visible=current.any().item(),common_pixels=int(np.count_nonzero((logits[0]>0)&(logits[1]>0))),tie_foreground_pixels=int(np.count_nonzero((logits[0]>0)&(logits[0]==logits[1])))))
     tables.append(dict(case=cid,offset=j,arm=name,ids=ids,actions=list(pairs.items()),S=s.tolist(),official=O.tolist()))
     if cid=='b1d7b3ac-0bdb47dc_149' and j==4:samples[name]=P.copy()
  print('SAME_MODEL_TWO_OWNERS',cid,flush=True)
with (OUT/'hard_example_masks.npz').open('xb') as file:np.savez_compressed(file,**samples)
save('actions.json',dict(records=records,matrices=tables,source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
