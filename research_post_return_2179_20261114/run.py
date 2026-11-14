from official import *
from assignment import edges,overlap_matrix
import gzip,torch
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from PIL import ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');v=p['video'];root=ROOT/'runs/z13_val20_matched_20261024/cache'/v;raw=np.load(root/'detections/raw_video.npz');clock={a['video_frame_index']:a for a in read(root/'alignment.json')['matches']};archive=json.load(gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz'));z=zipfile.ZipFile(ROOT/p['labelzip']);gt=labels(z,v);prev=ROOT/'research_official_candidates_2179_20261114';cand=read(prev/'candidate_result.json')['records'];mask=np.load(prev/'candidate_masks.npz');source=read(ROOT/'research_auto_prior_masks_20261114/protocol.json');ep=next(c for c in source['cases'] if c['id']=='old2179');folder=ROOT/ep['folder'];first=ep['frames'][0]['frame'];frames=p['frames'];checks=[];pred={};records=[];detail=[];panels=[];old_dec=read(ROOT/'research_mask_ownership_adapter_20261114/decision.json')['records'];old_own=read(ROOT/'research_assignment_weights_2179_20261112/runtime.json');e=next(q for q in old_own['events'] if q['frame']==1160);nearbox=next(q['reference']['box'] for q in e['rows'] if q['tid']==2445);w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt');cfg='configs/sam2.1/sam2.1_hiera_t.yaml'
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def propagate(mode,idx):
 state=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 if idx==0:
  for f in ep['ref']:vp.add_new_points_or_box(state,f['frame']-first,1,box=np.array(f['box'],'float32'))
 else:vp.add_new_points_or_box(state,1159-first,1,box=np.array(nearbox,'float32'))
 start=1159
 if mode=='confirmed' and idx==0:vp.add_new_mask(state,1171-first,1,mask['1171__111062']);start=1171
 result={}
 for j,_,s in vp.propagate_in_video(state,start_frame_idx=start-first,max_frame_num_to_track=frames[-1]-start):
  fr=first+j
  if fr in frames:result[fr]=(s[0,0]>0).cpu().numpy()
  state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
 return result
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);pr=[propagate('original',i) for i in [0,1]];re=propagate('confirmed',0);del vp;model=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False))
 for fr in frames:
  img=cv2.imread(str(folder/f'{fr-first:05d}.jpg'));model.set_image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));idx=clock[fr]['reference_index'];r=gt[idx];assert r['name']==clock[fr]['reference_name'];g=[masks(z,v,r,tid) for tid in p['official_ids']];ids=np.flatnonzero((raw['frame_idx']==fr)&(raw['cls']==2)&(raw['conf']>.1)).tolist();occ={a['raw_id']:a['tid'] for a in archive['assignments'] if a['frame']==fr and a['emitted']};rb=raw['xyxy'];occs=[rb[rid] for rid in occ if raw['cls'][rid]==2];reserved=occ.copy();ms=[]
  for rid in ids:
   if rid not in reserved and any(iou(rb[rid],b)>=.7 for b in occs):reserved[rid]='reserved_duplicate'
   if fr in (1165,1171):m=mask[f'{fr}__{rid}']
   else:m=model.predict(box=rb[rid].astype('float32'),multimask_output=False)[0][0].astype(bool)
   ms.append(m)
  overlaps=[[measure(m,*g[i])['iou'] for m in ms] for i in [0,1]];old_original=next((q for q in old_dec if q['case']=='old2179' and q['frame']==fr),None)
  highest=max([x or 0 for x in overlaps[0]],default=0);usable=[rid for rid,x in zip(ids,overlaps[0]) if (x or 0)>.5];already=next((rid for rid,tid in occ.items() if tid==2179),None)
  for method in ['original','confirmed']:
   pq=[(re[fr] if method=='confirmed' and fr>1171 else pr[0][fr]),pr[1][fr]];scores=overlap_matrix(pq,ms);choice=dict(edges(scores,[2179,2445],ids,protected=reserved).pairs)
   if old_original and method=='original':assert choice==dict(old_original['actions']['mask']['pairs'])
   composed=occ.copy()
   for tid,rid in choice.items():
    if tid not in composed.values():composed[rid]=tid
   assert all(composed[rid]==tid for rid,tid in occ.items()) and len(set(composed.values()))==len(composed);assigned=next((rid for rid,tid in composed.items() if tid==2179),None);m=ms[ids.index(assigned)] if assigned in ids else np.zeros(g[0][0].shape,bool);observed=measure(m,*g[0]);rawq=next((o for o in r['labels'] if o['id']==p['official_ids'][0]),None);excluded=rawq and any((rawq['attributes'] or {}).get(k,False) for k in ['ignored','ignore','crowd']);status='unknown_ignore' if excluded else 'wait' if assigned is None else 'correct' if (observed['iou'] or 0)>.5 else 'wrong'
   rec=dict(frame=fr,reference_index=idx,name=r['name'],method=method,near_owner=next((rid for rid,tid in composed.items() if tid==2445),None),action=assigned,native=already,visible=bool(g[0][0].any()),status=status,mask_iou=observed['iou'],prior_iou=measure(pq[0],*g[0])['iou'],best_iou=highest,truth_adequate_observations=usable,adequate_reserved=[rid for rid in usable if rid in reserved and reserved[rid]!=2179],original_pairs=len(occ),added=len(composed)-len(occ),prior_competitor=measure(pq[0],*g[1])['iou']);records.append(rec);detail.append(dict(frame=fr,method=method,ids=ids,original=list(occ.items()),reserved=list(reserved.items()),proposal=list(choice.items()),final=list(composed.items()),scores=scores.tolist(),target_claims=overlaps[0]));pred[f'{fr}__{method}__prior']=pq[0];pred[f'{fr}__{method}__chosen']=m
   if fr>1171 and method=='confirmed':
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);rgb[m]=(rgb[m]*.45+np.array([36,239,72])*.55).astype('uint8');target=g[0][0];co,_=cv2.findContours(target.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,co,-1,(255,213,20),2);ym,xm=np.where(target|pq[0]|m)
    if len(xm):x=max(0,xm.min()-35);y=max(0,ym.min()-25);xx=min(1280,xm.max()+36);yy=min(720,ym.max()+26)
    else:x,y,xx,yy=850,150,1280,500
    im=Image.fromarray(rgb[y:yy,x:xx]);im.thumbnail((335,180));panels.append((fr,status,im))
  print('EXACT_POST_RETURN',fr,flush=True)
with (OUT/'trajectory_masks.npz').open('xb') as f:np.savez_compressed(f,**pred)
save('actions.json',dict(records=records,matrices=detail,source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
sheet=Image.new('RGB',(1020,440),'#22272d');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for i,(fr,status,im) in enumerate(panels):
 x=(i%3)*340;y=(i//3)*220;sheet.paste(im,(x+(340-im.width)//2,y+30+(180-im.height)//2));draw.text((x+4,y+4),str(fr)+' · '+{'correct':'свой номер','wrong':'чужая рамка','wait':'ожидание','unknown_ignore':'нет оценки'}[status],fill='white',font=font)
sheet.save(OUT/'after_return.jpg',quality=92)
