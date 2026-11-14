from official import *
import torch
sys.path[:0]=[str(ROOT/'tools/sam2-runtime-20261027'),str(ROOT/'tools/sam2-deps-20261027')]
from tracking.gpu_lock import GpuLock
from sam2.build_sam import build_sam2_video_predictor
torch.set_num_threads(2);torch.set_default_device('cpu');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p=read(OUT/'protocol.json');db=read(OUT/'cases.json')['cases'];d=read(OUT/'detection_result.json');lab=zipfile.ZipFile(ROOT/p['labelzip']);cache={};rows=[];pred={}
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
 for q in [q for q in d['records'] if q['new_from_tiles']]:
  cid=q['case'];c=next(x for x in db if x['id']==cid);idx=c['index'];v=c['video']
  if v not in cache:cache[v]=labels(lab,v)
  folder=(ROOT/c['frames'][0]['image']).parent;state=model.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False);model.add_new_points_or_box(state,0,1,box=np.array(q['plus_tiles']['box'],'float32'))
  for j,_,logit in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=4):
   if j in [1,2,4]:
    m=(logit[0,0]>0).cpu().numpy();t,k,other=masks(lab,v,cache[v][idx+j],q['identity']);pred[f"{cid}__{q['slot']}__{j}"]=m;rows.append(dict(case=cid,video=v,split=q['split'],slot=q['slot'],offset=j,**measure(m,t,k,other)))
   state['output_dict_per_obj'][0]['non_cond_frame_outputs'].clear()
  del state;print('FOLLOW_RECOVERED',cid,q['slot'],flush=True)
with (OUT/'recovered_predictions.npz').open('xb') as f:np.savez_compressed(f,**pred)
stats={}
for split in ['familiar','fresh']:
 qs=[q for q in d['records'] if q['split']==split];vr=sorted({q['video'] for q in qs});delta=[np.mean([float(q['new_from_tiles']) for q in qs if q['video']==v]) for v in vr];a=np.array(delta);ii=np.random.default_rng(42).integers(len(vr),size=(10000,len(vr)));ci=np.quantile(a[ii].mean(1),[.025,.975]);rr=[x for x in rows if x['split']==split];stats[split]=dict(videos=len(vr),objects=len(qs),full=sum(q['full'] is not None for q in qs),added=sum(q['new_from_tiles'] for q in qs),added_allclass_full=sum(q['full'] is None and q['allclass_full'] is not None for q in qs),video_equal_delta=float(a.mean()),interval=ci.tolist(),recovered_future_visible=sum(x['iou'] is not None for x in rr),recovered_future_passing=sum(x['iou'] is not None and x['iou']>=.5 for x in rr),absent_future=sum(x['iou'] is None for x in rr))
fresh=stats['fresh'];valid=fresh['added']>0 and fresh['videos']==8 and fresh['interval'][0]>0 and fresh['recovered_future_passing']>0
save('result.json',dict(records=rows,stats=stats,scientific_passed=bool(valid),scope='Paired availability conditional on past groundtruth boxes, propagation on recovered boxes only. Existing boxes unchanged by construction. No estimate of unannotated objects, full tracker false tracks or overall tracking speed.'));print('MISSING_REPAIR',stats,flush=True)
