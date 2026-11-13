from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2,numpy as np,torch,time
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol_mask_refinement.json'); layout=read(PREV/'descriptor_plan.json');runtime=read(PREV/'runtime.json');old=np.load(PREV/'descriptors.npz',allow_pickle=False)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
masks=np.load(OUT/'mask_review/masks_reviewed.npz',allow_pickle=False)
roles={1159:('target_ref',next(r for e in runtime['events'] if e['frame']==1165 for r in e['rows'] if r['tid']==2179)['reference']['slot']),1164:('competitor_ref',next(r for e in runtime['events'] if e['frame']==1165 for r in e['rows'] if r['tid']==2445)['reference']['slot']),1165:('query',next(c for e in runtime['events'] if e['frame']==1165 for c in e['candidates'] if c['raw_id']==110260)['slot'])}
torch.set_num_threads(2);cv2.setNumThreads(2);arrays={};records=[];reproduction=[];start=time.monotonic()
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
    model=build('cuda');assert not model.training
    def fingerprint():return {n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items()}
    before=fingerprint()
    for frame,(role,slot) in roles.items():
        original=cv2.imread(str(OUT/'frames'/f'{frame}.png'));assert original.shape==(720,1280,3)
        groups=[masks[f'{frame}_{x}'].astype(bool) for x in ['target','neighbor','occluder','background','uncertain']]
        assert all(g.shape==original.shape[:2] for g in groups)
        assert np.all(np.stack(groups).sum(axis=0)==1),'Masks must partition the whole image'
        boxes=layout['frames'][slot[0]]['boxes'];offset=(slot[1]//16)*16;batch=np.asarray(boxes[offset:offset+16],np.float32);index=slot[1]-offset
        roi=torch.from_numpy(batch).to('cuda'); original_hash=hashlib.sha256(original.tobytes()).hexdigest()
        for fill in p['fills']:
            color=np.array([104,116,124] if fill=='imagenet_mean' else [0,0,0],np.uint8)
            for subset in p['subsets']:
                keep=np.zeros(original.shape[:2],bool)
                for bit,g in enumerate(groups):
                    if subset & (1<<bit):keep|=g
                modified=original.copy();modified[~keep]=color
                assert np.array_equal(modified[keep],original[keep])
                if subset==31:assert np.array_equal(modified,original)
                x,scale=preprocess(modified,'cuda');features=model.features(x);vectors=torch.nn.functional.normalize(model.embed_boxes(features,roi,scale),dim=1)
                vec=vectors[index].cpu().numpy().copy();assert vec.shape==(256,) and np.isfinite(vec).all()
                key=f'f{frame}_{fill}_{subset}';arrays[key]=vec;records.append(dict(key=key,frame=frame,role=role,subset=subset,fill=fill,slot=slot,roi_batch_size=len(batch),roi_index=index,retained_pixels=int(keep.sum())))
                if subset==31:
                    error=float(abs(vec-old[slot[0]][slot[1]]).max());reproduction.append(dict(frame=frame,fill=fill,max_error=error));assert error<1e-5,(frame,error)
                (OUT/'pixel_status.json').write_bytes((json.dumps(dict(updated=now(),completed=len(records),total=192))+'\n').encode())
                del x,features,vectors
        assert hashlib.sha256(original.tobytes()).hexdigest()==original_hash
    assert fingerprint()==before
with (OUT/'feature_outputs.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('feature_outputs.json',dict(passed=True,records=records,reproduction=reproduction,elapsed_seconds=time.monotonic()-start,mask_sha256=sha(OUT/'mask_review/masks_reviewed.npz'),bank_sha256=sha(OUT/'feature_outputs.npz'),model_unchanged=True,core_calls=0,tracker_calls=0,scope=p['scope']))
print('PIXEL_FACTORIAL_COMPLETE',len(records),flush=True)
