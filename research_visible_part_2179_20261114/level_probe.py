from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2,numpy as np,torch
from torchvision.ops import roi_align
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol_mask_refinement.json');layout=read(PREV/'descriptor_plan.json');rt=read(PREV/'runtime.json');mask=np.load(OUT/'mask_review/masks_reviewed.npz');prior=np.load(OUT/'feature_outputs.npz')
roles={1159:('target_ref',next(r for e in rt['events'] if e['frame']==1165 for r in e['rows'] if r['tid']==2179)['reference']['slot']),1164:('competitor_ref',next(r for e in rt['events'] if e['frame']==1165 for r in e['rows'] if r['tid']==2445)['reference']['slot']),1165:('query',next(c for e in rt['events'] if e['frame']==1165 for c in e['candidates'] if c['raw_id']==110260)['slot'])}
torch.set_num_threads(2);cv2.setNumThreads(2);arrays={};records=[]
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build('cuda');assert not model.training
 for fill in p['fills']:
  for mode in ['blank','full','red_query_clean_refs','black_query_clean_refs']:
   for frame,(role,slot) in roles.items():
    subset=0 if mode=='blank' else 31 if mode=='full' else 2 if role=='competitor_ref' or (role=='query' and mode=='black_query_clean_refs') else 1
    original=cv2.imread(str(OUT/'frames'/f'{frame}.png'));keep=np.zeros(original.shape[:2],bool)
    for bit,name in enumerate(['target','neighbor','occluder','background','uncertain']):
     if subset&(1<<bit):keep|=mask[f'{frame}_{name}']
    image=original.copy();image[~keep]=[104,116,124] if fill=='imagenet_mean' else [0,0,0]
    boxes=layout['frames'][slot[0]]['boxes'];offset=(slot[1]//16)*16;batch=torch.tensor(boxes[offset:offset+16],dtype=torch.float32,device='cuda');index=slot[1]-offset
    x,scale=preprocess(image,'cuda');features=model.features(x)
    for level in [-1,0,1]:
     if level<0:raw=model.embed_boxes(features,batch,scale)
     else:
      rois=torch.cat([torch.zeros((len(batch),1),device='cuda'),batch*scale],1)
      pooled=roi_align(features[level],rois,7,spatial_scale=1.0/(4*2**level),sampling_ratio=0,aligned=True)
      raw=model.embed(pooled)
     vec=torch.nn.functional.normalize(raw,dim=1)[index].cpu().numpy().copy();key=f'{fill}_{mode}_{role}_{level}';arrays[key]=vec;records.append(dict(key=key,fill=fill,mode=mode,role=role,level=level,frame=frame,subset=subset))
     if level<0:assert float(abs(vec-prior[f'f{frame}_{fill}_{subset}']).max())<1e-5
    (OUT/'level_status.json').write_bytes((json.dumps(dict(updated=now(),completed=len(records),total=72))+'\n').encode())
    del features,x
with (OUT/'level_features.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('level_features.json',dict(passed=True,records=records,source_sha256=sha(__file__),bank_sha256=sha(OUT/'level_features.npz'),core_calls=0,tracker_calls=0,scope='Fixed-input ROI pyramid-level diagnostic, cloned input only, no promotion. Native level -1 is identity control.'))
print('LEVEL_PROBE_COMPLETE',len(records))
