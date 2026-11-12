from common import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time,cv2,numpy as np,torch
from tracking.masa_r50 import build,preprocess
from tracking.gpu_lock import GpuLock
p=read(OUT/'protocol.json');assert read(OUT/'GO.json')['approved']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
torch.set_num_threads(2);cv2.setNumThreads(2);plan=read(OUT/'descriptor_plan.json');old=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/result.json');arrays={};receipts=[];start=time.monotonic();duplicate=False
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
    model=build('cuda');assert not model.training
    fingerprint=lambda:{n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items()}
    before=fingerprint();assert before==old['model_state']
    cap=cv2.VideoCapture(str(ROOT/plan['video_path']));assert cap.isOpened()
    for frame in range(max(map(int,plan['frames']))+1):
        ok,image=cap.read();assert ok,(frame,plan['video_path'])
        key=str(frame)
        if key not in plan['frames']:continue
        boxes=plan['frames'][key]['boxes'];x,scale=preprocess(image,'cuda');cpu,cs=preprocess(image,'cpu');assert scale==cs and torch.equal(x.cpu(),cpu);feature=model.features(x);parts=[]
        for j in range(0,len(boxes),16):
            batch=np.asarray(boxes[j:j+16],np.float32);first=not duplicate
            if first:batch=np.vstack([batch,batch[:1]])
            values=torch.nn.functional.normalize(model.embed_boxes(feature,torch.from_numpy(batch).to('cuda'),scale),dim=1)
            if first:assert torch.equal(values[0],values[-1]);values=values[:-1];duplicate=True
            parts.append(values.cpu().numpy().copy())
        array=np.concatenate(parts);assert array.shape==(len(boxes),256) and np.isfinite(array).all();arrays[key]=array
        receipts.append(dict(frame=frame,rois=len(boxes),pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        (OUT/'feature_status.json').write_bytes((json.dumps(dict(updated=now(),completed=len(receipts),total=len(plan['frames'])))+'\n').encode())
        del x,cpu,feature,values
    cap.release();assert duplicate and fingerprint()==before and len(arrays)==len(plan['frames'])
with (OUT/'descriptors.npz').open('xb') as f:np.savez_compressed(f,**arrays)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('feature_result.json',dict(passed=True,frames=receipts,rois=sum(len(a) for a in arrays.values()),bank_sha256=sha(OUT/'descriptors.npz'),elapsed_seconds=time.monotonic()-start,weights_unchanged=True,core_calls=0,tracker_calls=0,new_training=0,cpu_gpu_preprocess_exact=True,duplicate_exact=True,temp_environment={k:os.environ.get(k) for k in ['TEMP','TMP','TORCH_HOME','CUDA_CACHE_PATH']},gpu=torch.cuda.get_device_name()))
print('SMALL_DESCRIPTOR_BANK_COMPLETE',len(arrays),sum(map(len,arrays.values())),flush=True)
