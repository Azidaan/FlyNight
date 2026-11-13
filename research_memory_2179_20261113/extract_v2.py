from common import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time, cv2, numpy as np, torch
from tracking.masa_r50 import build, preprocess
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol.json'); plan=read(OUT/'descriptor_plan.json')
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
torch.set_num_threads(2); cv2.setNumThreads(2)
arrays={}; receipts=[]; started=time.monotonic()
old_model=read(PREV/'feature_result.json')
old_plan=read(PREV/'descriptor_plan.json'); old_bank=np.load(PREV/'descriptors.npz',allow_pickle=False)
overlap_error=0.; overlap_count=0
with GpuLock('Local\\DissertationSingleGpu'), torch.inference_mode():
    model=build('cuda'); assert not model.training
    def fingerprint(): return {n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items()}
    before=fingerprint(); assert before==read(ROOT/'research_remaining_causes_20261112/independent_review/reid/result.json')['model_state']
    cap=cv2.VideoCapture(str(ROOT/plan['video_path'])); assert cap.isOpened()
    for frame in range(max(map(int,plan['frames']))+1):
        ok,image=cap.read(); assert ok,frame
        key=str(frame)
        if key not in plan['frames']: continue
        boxes=plan['frames'][key]['boxes']; image_tensor,scale=preprocess(image,'cuda'); feature=model.features(image_tensor); parts=[]
        template=old_plan['frames'].get(key,{}).get('boxes',boxes)
        for box in boxes:
            # Match the original ROI-head batch shape and position. For a raw
            # counterpart substitute only its own filtered box in that batch.
            match=box if box in template else None
            if match is None:
                for obs in plan['observations']:
                    if obs['frame']==frame and obs['raw']==box and obs['filtered'] in template:
                        match=obs['filtered']; break
            position=template.index(match) if match is not None else 0
            offset=(position//16)*16; batch=np.asarray(template[offset:offset+16],np.float32).copy(); local=position-offset
            batch[local]=np.asarray(box,np.float32)
            values=torch.nn.functional.normalize(model.embed_boxes(feature,torch.from_numpy(batch).to('cuda'),scale),dim=1)
            a=values[local].cpu().numpy().copy(); parts.append(a)
            if key in old_plan['frames'] and box in old_plan['frames'][key]['boxes']:
                k=old_plan['frames'][key]['boxes'].index(box)
                overlap_error=max(overlap_error,float(abs(a-old_bank[key][k]).max())); overlap_count+=1
        a=np.stack(parts); assert a.shape==(len(boxes),256) and np.isfinite(a).all(); arrays[key]=a
        receipts.append(dict(frame=frame,rois=len(boxes),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        (OUT/'feature_status_v2.json').write_bytes((json.dumps(dict(updated=now(),completed=len(receipts),total=len(plan['frames'])))+'\n').encode())
    cap.release(); assert fingerprint()==before
assert overlap_count>0 and overlap_error<1e-5,(overlap_count,overlap_error)
with (OUT/'descriptors.npz').open('xb') as f: np.savez_compressed(f,**arrays)
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
save('feature_result.json',dict(passed=True,frames=receipts,bank_sha256=sha(OUT/'descriptors.npz'),elapsed_seconds=time.monotonic()-started,old_descriptor_overlap_count=overlap_count,old_descriptor_max_error=overlap_error,model_weights_unchanged=True,training_calls=0,core_calls=0,tracker_calls=0))
print('HISTORY_FEATURES_READY',len(receipts),sum(len(x) for x in arrays.values()),flush=True)
