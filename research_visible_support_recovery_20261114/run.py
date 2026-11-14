from official import *
from assignment import edges
from collections import Counter
p=read(OUT/'protocol.json');parent=ROOT/'research_mask_ownership_adapter_20261114';tr=read(parent/'decision.json')['records'];a=np.load(parent/'prior.npz');b=np.load(parent/'observations.npz');official=read(parent/'result.json');last=ROOT/'research_ownership_conflict_2179_20261114';source=read(last/'measurements.json');db=np.load(last/'mismatch_inputs.npz');own=read(ROOT/'research_post_return_2179_20261114/actions.json');rows=[];matrices=[];old=[];z=zipfile.ZipFile(ROOT/p['labelzip']);cases={x['id']:x for x in read(ROOT/'research_depth_identity_competition_20261114/pairs.json')['cases']};cache={};diagnostic=None
def support(a,b):
 tp=int((a&b).sum());sa=int(a.sum());sb=int(b.sum());return [tp/max(1,sa+sb-tp),tp/max(1,sa),tp/max(sa,sb,1)]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for q in tr:
 cid=q['case'];j=q['frame'];owners=q['owners'];ids=[r['id'] for r in q['items']];ms=[b[f'{cid}__{j}__{i}'] for i in ids];prior=[a[f'{cid}__{j}__{i}'] for i in (0,1)];cube=np.array([[support(o,m) for m in ms] for o in prior]).reshape(2,len(ids),3);original=dict(q['actions']['mask']['pairs']);assert np.allclose(cube[:,:,0],np.asarray(q['actions']['mask']['scores']),rtol=0,atol=1e-12);reserve=dict(q['reserved']);actual=dict(q['protected']);none=np.zeros((720,1280),bool);v=q['video']
 if cid!='old2179':
  if v not in cache:cache[v]=labels(z,v)
  c=cases[cid];g=[masks(z,v,cache[v][c['index']+j],o) for o in c['ids']]
 for i,name in enumerate(['jaccard','prior_coverage','both_coverages']):
  S=cube[:,:,i];cand=dict(edges(S,owners,ids,protected=reserve).pairs);out=original.copy()
  for tid,rid in cand.items():
   if tid not in out and rid not in out.values():out[tid]=rid
  matrices.append(dict(case=cid,frame=j,name=name,scores=S.tolist(),ids=ids,owners=owners,reserved=list(reserve.items()),proposed=list(cand.items()),baseline=list(original.items()),combined=list(out.items())))
  if cid=='old2179':
   old.append(dict(frame=j,method=name,far=out.get(2179),near_original=[rid for rid,tid in actual.items() if tid==2445]));continue
  for ii,tid in enumerate(owners):
   rid=out.get(tid);m=ms[ids.index(rid)] if rid is not None else none;io=measure(m,*g[ii])['iou'];st='deferred' if rid is None else 'correct' if (io or 0)>.5 else 'wrong'
   if name=='jaccard':
    orig=next(x for x in official['transfer_rows'] if x['case']==cid and x['frame']==j and x['slot']==ii and x['method']=='mask');assert st==orig['status'] and rid==orig['action']
   rows.append(dict(case=cid,video=v,frame=j,slot=ii,name=name,status=st,action=rid,iou=io))
# Full candidate and rival denominator on1176 needed to decide an actual action; do not extrapolate from a single pair.
from tracking.gpu_lock import GpuLock
import torch
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rt=read(ROOT/'research_assignment_weights_2179_20261112/runtime.json');e=next(q for q in rt['events'] if q['frame']==1160);box=next(q['reference']['box'] for q in e['rows'] if q['tid']==2445);v=read(ROOT/'research_post_return_2179_20261114/protocol.json')['video'];raw=np.load(ROOT/f'runs/z13_val20_matched_20261024/cache/{v}/detections/raw_video.npz');oq=next(q for q in own['matrices'] if q['frame']==1176 and q['method']=='confirmed');ids=oq['ids'];cfg='configs/sam2.1/sam2.1_hiera_t.yaml';w=str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt')
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 vp=build_sam2_video_predictor(cfg,w,device='cuda',apply_postprocessing=False);st=vp.init_state(str(ROOT/'research_mask_propagation_20261114/frames'),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);vp.add_new_points_or_box(st,0,1,box=np.array(box,'float32'))
 for j,_,val in vp.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=17):
  if j==17:near=(val[0,0]>0).cpu().numpy()
  st['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
 del vp,st;im=cv2.imread(str(ROOT/'research_mask_propagation_20261114/frames/00017.jpg'));net=SAM2ImagePredictor(build_sam2(cfg,w,device='cuda',apply_postprocessing=False));net.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ms=[net.predict(box=raw['xyxy'][rid].astype('float32'),multimask_output=False)[0][0].astype(bool) for rid in ids];assert np.array_equal(ms[ids.index(111813)],db['current'])
 cube=np.array([[support(x,m) for m in ms] for x in [db['prior'],near]]);assert np.allclose(cube[:,:,0],oq['scores'],rtol=0,atol=1e-12);diagnostic={}
 for i,name in enumerate(['jaccard','prior_coverage','both_coverages']):
  S=cube[:,:,i];pairs=dict(edges(S,[2179,2445],ids,protected=dict(oq['reserved'])).pairs);diagnostic[name]=dict(far=pairs.get(2179),candidate_score=S[0,ids.index(111813)],best_score=S[0].max(),best_raw=ids[int(S[0].argmax())],source_correct_when_proposed=pairs.get(2179)==111813)
save('actions.json',dict(rows=rows,matrices=matrices,old=old,case1176=diagnostic))
