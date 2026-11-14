from official import *
import torch
from scipy.optimize import linear_sum_assignment
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_yolo26seg');os.environ['YOLO_AUTOINSTALL']='false'
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');last=ROOT/'research_natural_gap_ownership_20261114';cases=read(last/'cases.json')['cases'];previous=read(last/'actions.json');rows=[];diagnosis=[];ref_vectors={};sims={};traces=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def detections(im):
 d=det.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,half=False,batch=1,device=0,verbose=False,save=False)[0];return [(i,b,float(sc)) for i,(b,sc) in enumerate(zip(d.boxes.xyxy.cpu().numpy(),d.boxes.conf.cpu().numpy()))]
def embeddings(im,boxes):
 a,scale=preprocess(im,'cuda');features=model.features(a);b=torch.tensor(np.array(boxes).reshape(-1,4),dtype=torch.float32,device='cuda');vec=model.embed_boxes(features,b,scale);return torch.nn.functional.normalize(vec,dim=1).cpu().numpy()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build('cuda');det=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
 for c in cases:
  cid=c['id'];fs=c['frames'];im=cv2.imread(str(ROOT/fs[0]['image']));obs=detections(im);truth=c['past_boxes'];mat=np.array([[iou(np.array(a),bb) for _,bb,_ in obs] for a in truth]);obj=np.where(mat>=.5,1+mat,0);ii,jj=linear_sum_assignment(-np.c_[obj,np.zeros((2,2))]);box=[None,None]
  for i,j in zip(ii,jj):
   if j<len(obs) and obj[i,j]>0:box[i]=obs[j][1]
  maskbase=previous['records'];exist=[b is not None for b in box]
  for i in [0,1]:assert exist[i]==next(x for x in maskbase if x['case']==cid and x['slot']==i)['prior_available']
  index=[i for i in [0,1] if box[i] is not None];v0=embeddings(im,[box[i] for i in index]);prior=np.zeros((2,256),'float32')
  for pos,i in enumerate(index):prior[i]=v0[pos]
  ref_vectors[cid+'__past']=prior
  for j,f in enumerate(fs[1:],1):
   im=cv2.imread(str(ROOT/f['image']));pool=[x for x in detections(im) if x[2]>.1];ids=[rid for rid,_,_ in pool];cur=embeddings(im,[b for _,b,_ in pool]);cos=prior.astype('float64')@cur.astype('float64').T;ref_vectors[f'{cid}__{j}']=cur;tr=next(x for x in previous['traces'] if x['case']==cid and x['frame']==j);assert ids==tr['rids'];gt=np.asarray(tr['overlap_with_truth']);selection={}
   for i in index:
    if len(ids)==0:continue
    k=int(np.argmax(cos[i]));sim=cos[i,k]
    # .5 cosine cutoff is fixed beforehand and never tuned; source-available rows only.
    if sim>.5 and (sim>np.delete(cos[i],k)).all() and all(sim>cos[z,k] for z in index if z!=i):selection[i]=ids[k]
   base=dict(tr['actions']['mask']);combine=base.copy()
   for i,rid in selection.items():
    if i not in combine and rid not in combine.values():combine[i]=rid
   for mode,act in [('mask',base),('appearance',selection),('fill_missing',combine)]:
    assert len(act.values())==len(set(act.values()))
    for i in [0,1]:
     orig=next(x for x in maskbase if x['case']==cid and x['slot']==i and x['frame']==j and x['method']=='mask');rid=act.get(i);u=gt[i,ids.index(rid)] if rid is not None else None;st='unknown_ignore' if orig['ignored'] else 'wait' if rid is None else 'correct' if u>.5 else 'wrong'
     if mode=='mask':assert st==orig['status']
     rows.append(dict(case=cid,video=c['video'],frame=j,slot=i,mode=mode,return_frame=orig['first_return'],status=st,raw_id=rid,target_iou=float(u) if u is not None else None,available=orig['available_current'],source_available=orig['prior_available']))
   if j==len(fs)-1:
    for i in range(2):
     good=[k for k in range(len(ids)) if gt[i,k]>.5];sort=sorted(range(len(ids)),key=lambda k:(-cos[i,k],ids[k]));rank=next((h+1 for h,k in enumerate(sort) if k in good),None) if exist[i] else None;diagnosis.append(dict(case=cid,video=c['video'],slot=i,admissible_correct_ids=[ids[k] for k in good],source_available=exist[i],first_correct_rank=rank,chosen_by_appearance=selection.get(i),chosen_by_mask=base.get(i),chosen_combined=combine.get(i),cosine_best=float(max(cos[i],default=0)) if exist[i] else None))
   traces.append(dict(case=cid,frame=j,ids=ids,cos=cos.tolist(),observed_rows=index,appearance=list(selection.items()),mask=list(base.items()),combined=list(combine.items())))
  print('APPEARANCE_ON_GAP',cid,flush=True)
with (OUT/'vectors.npz').open('xb') as file:np.savez_compressed(file,**ref_vectors)
save('actions.json',dict(records=rows,ranks=diagnosis,traces=traces))
