"""Реальные изображения истории и фиксированные признаки для1/3/4; безGT/обучения."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    import numpy as np,cv2,torch,shutil
    cv2.setNumThreads(1)
    from tracking.gpu_lock import GpuLock
    from tracking.masa_r50 import build
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.causal_motion_history import CausalMotionHistory
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    folder=RUN/'history_visual_inputs';p=read(folder/'protocol.json');end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    assert shutil.disk_usage(ROOT).free>=p['minimum_free_bytes']
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    previous=RUN/'history_pair_audit';c=read(previous/'controller_result.json');audit=read(previous/'report.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert audit['passed'] and audit['full_dataset']
    check=RUN/'history_visual_checks';assert read(check/'report.json')['passed']
    c=read(check/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    torch.set_num_threads(2);completed=[]
    raw_sources={x['video']:x for x in read(RUN/'history_observed_export/report.json')['completed']}
    with GpuLock('Local\\DissertationSingleGpu'):
        encoder=build();encoder.eval();buffers={k:v.clone() for k,v in encoder.named_buffers()}
        dataset=read(RUN/'history_pair_dataset/report.json')
        for vi,entry in enumerate(dataset['completed']):
            started=time.monotonic();v=entry['video'];out=folder/'videos'/v;out.mkdir(parents=True,exist_ok=True)
            if (out/'report.json').exists():
                m=read(out/'report.json');assert all(sha(out/k)==h for k,h in m['sha256'].items());completed.append(m);continue
            inputs=RUN/'history_pair_dataset/videos'/v/'inputs.npz';assert sha(inputs)==entry['inputs_sha256']
            with np.load(inputs) as z:x={k:z[k] for k in z.files}
            keys={(int(f),int(d)) for f,d in zip(x['frame'],x['det'])}
            keys.update((int(f),int(d)) for f,d,valid in zip(x['history_frame'].ravel(),x['history_det'].ravel(),x['history_valid'].ravel()) if valid)
            ordered=sorted(keys);lookup={k:i for i,k in enumerate(ordered)};n=len(ordered)
            expected_boxes={}
            for frame,det,box in zip(x['frame'],x['det'],x['box']):expected_boxes[(int(frame),int(det))]=box
            for frame,det,box,valid in zip(x['history_frame'].ravel(),x['history_det'].ravel(),x['history_box'].reshape(-1,4),x['history_valid'].ravel()):
                if valid:
                    key=(int(frame),int(det))
                    if key in expected_boxes:assert np.array_equal(expected_boxes[key],box)
                    expected_boxes[key]=box
            wanted={}
            for i,(frame,det) in enumerate(ordered):wanted.setdefault(frame,[]).append((i,det))
            current_index=np.array([lookup[(int(f),int(d))] for f,d in zip(x['frame'],x['det'])],np.int32)
            history_index=np.full(x['history_frame'].shape,-1,np.int32)
            for i,j in zip(*np.nonzero(x['history_valid'])):history_index[i,j]=lookup[(int(x['history_frame'][i,j]),int(x['history_det'][i,j]))]
            embeddings=np.empty((n,256),np.float32);images=np.empty((n,3,p['image_size'],p['image_size']),np.uint8)
            camera=np.empty((n,2),np.float32);epochs=np.empty(n,np.int32)
            video_path=ROOT/f'data/bdd100k/videos/train/{v}.mov';video_hash=sha(video_path)
            assert video_hash==read(ROOT/f'runs/train_cache/{v}/detections/raw_video.json')['identity']['video_sha256']
            capture=cv2.VideoCapture(str(video_path));assert capture.isOpened()
            cache_path=ROOT/f'runs/train_cache/{v}/detections/raw_video.npz';assert sha(cache_path)==raw_sources[v]['source_sha256']
            cache=load_detection_cache(cache_path)
            schedule={int(r['video_frame_index']) for r in read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches']}
            motion=CausalMotionHistory(**p['motion']);server=FrameMasaServer(encoder);failed_camera=0;seen=0
            for frame in range(max(schedule)+1):
                if time.time()>end-15:raise TimeoutError('History visual cache original budget')
                ok,image=capture.read();assert ok
                if frame not in schedule:continue
                state=motion.update(image,frame);failed_camera+=not state['reliable']
                if frame in wanted:
                    b,_,_=detections_for_frame(cache,frame);items=wanted[frame]
                    assert all(np.array_equal(b[d],expected_boxes[(frame,d)]) for i,d in items)
                    server.start_frame(frame,image);tokens=[server.observation(b[d]) for i,d in items];vectors=server.features(tokens)
                    for (i,d),vector in zip(items,vectors):
                        assert np.isfinite(vector).all()
                        embeddings[i]=vector;images[i]=motion.crop(b[d],image.shape,p['image_size'],p['context'])
                        camera[i]=state['camera'];epochs[i]=state['epoch'];seen+=1
                    server.finish_frame()
                write(folder/'status.json',dict(state='running',video=v,index=vi,total=len(dataset['completed']),frame=frame,rois_done=seen,rois_total=n,updated=now()),required=False)
            capture.release();assert seen==n and all(torch.equal(buffers[k],value) for k,value in encoder.named_buffers())
            assert np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-5)
            np.save(out/'embeddings.npy',embeddings);np.save(out/'images.npy',images)
            np.savez_compressed(out/'index.npz',roi_frame=np.array([k[0] for k in ordered],np.int32),roi_det=np.array([k[1] for k in ordered],np.int32),current_index=current_index,history_index=history_index,camera=camera,camera_epoch=epochs,video_size=np.array([image.shape[1],image.shape[0]],np.int32))
            m=dict(video=v,split=entry['split'],rois=n,pairs=entry['pairs'],failed_camera_samples=int(failed_camera),seconds=time.monotonic()-started,
                input_sha256=sha(inputs),video_sha256=video_hash,sha256={k:sha(out/k) for k in ['embeddings.npy','images.npy','index.npz']},gt_read=False,weights_trained=False)
            write(out/'report.json',m);completed.append(m);write(folder/'completed_videos.json',completed)
            if vi==0:
                need=2*m['seconds']*(len(dataset['completed'])-1)+120;available=end-time.time();price=dict(first_full_video_seconds=m['seconds'],conservative_remaining_seconds=need,available_seconds=available,proceed=need<available);write(folder/'price.json',price)
                if not price['proceed']:
                    write(folder/'report.json',dict(complete=False,stopped_on_cost=True,completed=completed,remaining=dataset['completed'][1:],training_ready=False,hota_measured=False,updated=now()));return
        write(folder/'report.json',dict(complete=True,completed=completed,rois=sum(x['rois'] for x in completed),weights_frozen=True,gt_read=False,training_ready=False,hota_measured=False,updated=now()))

if __name__=='__main__':main()
