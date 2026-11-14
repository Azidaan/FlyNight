from official import *
import torch
from torchvision.ops import roi_align
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
torch.set_num_threads(2);cv2.setNumThreads(2);p=read(OUT/'protocol.json');truth=np.load(OUT/'official_masks.npz');rt=read(ROOT/'research_assignment_weights_2179_20261112/runtime.json');e=next(r for r in rt['events'] if r['frame']==1160);prior={name:next(r for r in e['rows'] if r['tid']==tid)['reference']['box'] for tid,name in [(2179,'far'),(2445,'near')]};arrays={};records=[]
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build('cuda')
 for fr in p['frames']:
  im=cv2.imread(str(ROOT/f'research_mask_propagation_20261114/frame_{fr}.png'));bb={}
  for role in ['far','near']:
   ys,xs=np.where(truth[f'{fr}__{role}']);bb[role]=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
  mixed=[min(b[0] for b in bb.values()),min(b[1] for b in bb.values()),max(b[2] for b in bb.values()),max(b[3] for b in bb.values())]
  for role in ['far','near']:
   for fill,color in [('mean',[104,116,124]),('black',[0,0,0])]:
    for mode in ['full','masked','blank']:
     picture=im.copy()
     if mode=='masked':picture[~truth[f'{fr}__{role}']]=color
     if mode=='blank':picture[:]=color
     image,scale=preprocess(picture,'cuda');ft=model.features(image)
     for roi,box in [('own',prior[role] if fr==1159 else bb[role]),('mixed',prior[role] if fr==1159 else mixed)]:
      batch=torch.tensor([box],device='cuda',dtype=torch.float32)
      for level in [-1,0]:
       if level==-1:out=model.embed_boxes(ft,batch,scale)
       else:
        rois=torch.cat([torch.zeros((1,1),device='cuda'),batch*scale],1);out=model.embed(roi_align(ft[0],rois,7,spatial_scale=1/4,sampling_ratio=0,aligned=True))
       arr=torch.nn.functional.normalize(out,dim=1)[0].cpu().numpy();arrays[f'{fr}__{role}__{roi}__{mode}__{fill}__{level}']=arr
     del ft,image
  print('APPEARANCE_FRAME',fr,flush=True)
for fr in [1165,1171]:
 for roi in ['own','mixed']:
  for mode in ['full','masked','blank']:
   for fill in ['mean','black']:
    for lev in [-1,0]:
     get=lambda f,role:arrays[f'{f}__{role}__{roi}__{mode}__{fill}__{lev}'].astype('float64')
     c=np.array([[1-np.dot(get(1159,i),get(fr,j)) for j in ['far','near']] for i in ['far','near']]);margin=float(c[0,1]+c[1,0]-c[0,0]-c[1,1]);records.append(dict(frame=fr,roi=roi,mode=mode,fill=fill,level=lev,costs=c.tolist(),global_margin=margin,correct=bool(margin>0),local_far=bool(c[0,0]<c[1,0]),local_near=bool(c[1,1]<c[0,1])))
with (OUT/'vectors.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('appearance_result.json',dict(records=records,vector_count=len(arrays),scope='Official masks and boxes privileged. Same-past1159 for both, replacing problematic1164 without direct official labels; not exact redo of old protocol. Small model-only margin, no ID switch implication.'))
