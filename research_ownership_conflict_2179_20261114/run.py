from official import *
from assignment import edges,overlap_matrix
import gzip,torch
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.gpu_lock import GpuLock
from PIL import ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');src=ROOT/'research_post_return_2179_20261114';orig=read(src/'actions.json');vp=np.load(src/'trajectory_masks.npz');v=p['video'];folder=ROOT/'research_early_target_parts_20261114/frames';raw=np.load(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz');labelsdb=zipfile.ZipFile(ROOT/p['labelzip']);allgt=labels(labelsdb,v);clock={x['video_frame_index']:x for x in read(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/alignment.json')['matches']};hist=json.load(gzip.open(ROOT/f'runs/domain_matched_native_20261109/rollouts/C34/{v}.events.json.gz'));e=read(ROOT/'research_assignment_weights_2179_20261112/runtime.json');old=read(ROOT/'research_official_candidates_2179_20261114/candidate_result.json')['records'];data=[];raster={};ledger=[];panels=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def sims(a,b,k):return float((a&b&k).sum()/max(1,(k&(a|b)).sum()))
def official(vv,r):
 out=[]
 for ob in r['labels']:
  if any((ob.get('attributes') or {}).get(h,False) for h in ['ignored','ignore','crowd','iscrowd']):continue
  ma=rle(ob);out.append((ob['id'],ob['category'],ma))
 return out
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False))
 for fr in p['frames']:
  idx=clock[fr]['reference_index'];r=allgt[idx];assert r['name']==clock[fr]['reference_name'];g=[masks(labelsdb,v,r,ident) for ident in p['official_ids']];gt,k,other=g[0];near=g[1][0];pool=np.flatnonzero((raw['frame_idx']==fr)&(raw['cls']==2)&(raw['conf']>.1)).tolist();im=cv2.cvtColor(cv2.imread(str(folder/f'{fr-1141:05d}.jpg')),cv2.COLOR_BGR2RGB);model.set_image(im);ms=[model.predict(box=raw['xyxy'][rid].astype('float32'),multimask_output=False)[0][0].astype(bool) for rid in pool];allobs=official(v,r);owners={x['raw_id']:x['tid'] for x in hist['assignments'] if x['frame']==fr and x['emitted']};d=next((q for q in orig['matrices'] if q['frame']==fr and q['method']=='confirmed'),None)
  for rid,m in zip(pool,ms):
   if rid not in owners or owners[rid] not in [2179,2445,2777]:continue
   best=sorted([dict(id=i,category=c,iou=sims(m,mask,k)) for i,c,mask in allobs],key=lambda q:(-q['iou'],q['id']))[:3];ledger.append(dict(frame=fr,rid=rid,tid=owners[rid],top=best,agrees_far=best and best[0]['id']==p['official_ids'][0] and best[0]['iou']>.5,agrees_near=best and best[0]['id']==p['official_ids'][1] and best[0]['iou']>.5))
  if d is None:continue
  assert pool==d['ids'];prior=vp[f'{fr}__confirmed__prior'];nearS=np.array(d['scores'][1]);farS=np.array([sims(prior,m,np.ones(k.shape,bool)) for m in ms]);S=np.stack([farS,nearS]);assert np.allclose(S,d['scores'],rtol=0,atol=1e-12);reserved=dict(d['reserved']);real=dict(d['original']);controlled={n:dict(edges(S,[2179,2445],pool,protected=oc).pairs) for n,oc in [('registered',reserved),('owners_without_duplicates',real),('remove_2445_only',{j:t for j,t in reserved.items() if t!=2445}),('remove_2777_only',{j:t for j,t in reserved.items() if t!=2777}),('unprotected',{})]}
  valid_scores=np.stack([[sims(prior,m,k) for m in ms],nearS]);# Secondary diagnostic changes only the far row using privileged ignore map.
  ignore=dict(edges(valid_scores,[2179,2445],pool,protected=reserved).pairs)
  fits=[]
  for rid,m in zip(pool,ms):
   jf=sims(m,gt,k);jn=sims(m,near,k);rawsim=farS[pool.index(rid)];valsim=valid_scores[0,pool.index(rid)];inter=int((prior&m).sum());union=int((prior|m).sum());ignored=int(((m|prior)&~k).sum());lost=int((m&~k).sum())
   if jf>.5 or rawsim>.5 or rid==111813:
    fits.append(dict(rid=rid,owner=real.get(rid),reservation=reserved.get(rid),target_iou=jf,neighbor_iou=jn,comparison_unmasked=float(rawsim),comparison_valid_only=float(valsim),intersection=inter,union=union,ignored_union=ignored,ignored_predicted=lost,mask_size=int(m.sum()),prior_size=int(prior.sum()),valid_target=int(gt.sum()),other_ignored_ids=sorted([dict(id=o['id'],n=int((rle(o)&m&~k).sum())) for o in r['labels'] if (o['attributes'] or {}).get('ignored',False) and np.any(rle(o)&m&~k)],key=lambda q:-q['n'])[:4]))
    if rid==111813:raster['prior']=prior;raster['current']=m;raster['valid']=k;raster['target']=gt
  def outcomes(choices):
   return {str(i):dict(rid=j,iou_to_expected=sims(ms[pool.index(j)],g[0 if i==2179 else 1][0],k)) for i,j in choices.items()}
  data.append(dict(frame=fr,rows= fits,controlled={n:outcomes(vals) for n,vals in controlled.items()},privileged_ignore_far_only=outcomes(ignore),actual_owners=sorted((rid,tid) for rid,tid in real.items() if tid in [2179,2445,2777])))
  if fr in [1176,1182,1206]:
   m=ms[max(range(len(ms)),key=lambda ix:sims(ms[ix],gt,k))];shown=im.copy();shown[m]=(shown[m]*.48+np.array([40,232,66])*.52).astype('uint8');cont,_=cv2.findContours(gt.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(shown,cont,-1,(250,212,24),2);iy,ix=np.where(m|gt);patch=Image.fromarray(shown[max(0,iy.min()-25):min(720,iy.max()+26),max(0,ix.min()-30):min(1280,ix.max()+31)]);patch.thumbnail((390,220));panels.append((fr,patch))
with (OUT/'mismatch_inputs.npz').open('xb') as f:np.savez_compressed(f,**raster)
save('measurements.json',dict(records=data,owner_ledger=ledger,source_unchanged=all(sha(ROOT/n)==h for n,h in p['inputs'].items())))
sheet=Image.new('RGB',(1200,270),'#23282c');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
for i,(fr,patch) in enumerate(panels):sheet.paste(patch,(i*400+(400-patch.width)//2,37+(220-patch.height)//2));dr.text((i*400+5,7),{1176:'1176: свободное наблюдение',1182:'1182: рамка занята 2445',1206:'1206: рамка занята 2777'}[fr],font=font,fill='white')
sheet.save(OUT/'conflicts.jpg',quality=92)
