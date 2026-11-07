"""Короткая проверка причинного MASA-входа в реальные назначения, не HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now

def main():
    import cv2,numpy as np,torch
    from tracking.masa_r50 import build,load_state_dict,_map_key,preprocess
    from tracking.frame_masa_fp16_appearance import FrameMasaServer
    from tracking.lazy_appearance_association import LazyAppearanceXYWH
    from tracking.offline_association import _DetSlice
    from tracking.gpu_lock import GpuLock
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    run=BASE/'frame_masa_fp16_profile';p=read(run/'protocol.json');end=datetime.fromisoformat(read(run/'execution_deadline.json')['deadline']).timestamp();torch.set_num_threads(2)
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    source=p['source'];cache=load_detection_cache(ROOT/source['detections']);began=time.monotonic()
    with GpuLock('Local\\DissertationSingleGpu'):
        model=build('cuda');buffers={k:v.clone() for k,v in model.named_buffers()};server=FrameMasaServer(model)
        tracker=LazyAppearanceXYWH(SimpleNamespace(**p['config']),server,weight=.1)
        cap=cv2.VideoCapture(str(ROOT/source['video_path']));assert cap.isOpened();torch.cuda.reset_peak_memory_stats();samples=[];numeric_pairs=[]
        for frame in range(p['frames']):
            if time.time()>end-5:raise TimeoutError('Original MASA profile deadline')
            ok,image=cap.read();assert ok;server.start_frame(frame,image);xyxy,conf,classes=detections_for_frame(cache,frame)
            tracker.update(_DetSlice(xyxy,conf,classes));server.finish_frame()
            current=[t for t in server.tokens.values() if t.frame==frame]
            assert all(t.vector is not None and np.isfinite(t.vector).all() for t in current)
            if frame in [0,119] and current:
                with torch.inference_mode():
                    image_tensor,scale=preprocess(image,'cuda');boxes=torch.from_numpy(np.stack([t.crop for t in current[:2]])).to('cuda')
                    reference=torch.nn.functional.normalize(model.embed_boxes(model.features(image_tensor),boxes,scale),dim=1).cpu().numpy()
                computed=np.stack([t.vector for t in current[:2]]);cos=(reference*computed).sum(1)/(np.linalg.norm(reference,axis=1)*np.linalg.norm(computed,axis=1))
                numeric_pairs.append(dict(frame=frame,cosine=cos.tolist(),max_absolute_difference=float(np.max(np.abs(reference-computed)))))
                assert np.isfinite(cos).all() and float(cos.min())>=.99
            if frame in [0,1,119]:samples.append(dict(frame=frame,n=len(current),vectors=[t.vector[:8].tolist() for t in current[:2]]))
            write(run/'status.json',dict(state='running',updated=now(),frame=frame,total=p['frames'],embedded=server.embedded,encoded_frames=server.encoded_frames),required=False)
        cap.release();torch.cuda.synchronize();assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
        sd=load_state_dict();own=model.state_dict();mapped={_map_key(k):v for k,v in sd.items() if _map_key(k) in own and own[_map_key(k)].shape==v.shape}
        missing=[k for k in own if k not in mapped and not k.endswith('num_batches_tracked')];assert not missing
        seconds=time.monotonic()-began
        write(run/'report.json',dict(passed=True,frames=p['frames'],embedded=server.embedded,encoded_frames=server.encoded_frames,seconds_including_load=seconds,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),mapped_keys=len(mapped),required_state_keys=len(own),missing_required=missing,
            checkpoint_sha256=sha(ROOT/'weights/masa/masa_r50.pth'),strict_past=all(all(f<e['frame'] for f in e['past_frames']) for e in tracker.events),
            finite=True,numeric_pairs=numeric_pairs,numeric_precision='mixedFP16_float32_similarity',batchnorm_unchanged=True,events=len(tracker.events),samples=samples,
            limitation='Технические120кадров,не полное видео и не HOTA. Цена полной серии требует первого полного видео. Своя интеграцияMASAвпрежнийтрекер,неполныйавторскийтрекер.'))
    write(run/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':main()
