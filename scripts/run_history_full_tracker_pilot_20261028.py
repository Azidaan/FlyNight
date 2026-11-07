"""Полные32трекера на40историческихval; все входы только текущие/прошлые."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def main():
    import numpy as np,torch,cv2
    from tracking.gpu_lock import GpuLock
    from tracking.masa_r50 import build
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.causal_motion_history import CausalMotionHistory
    from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead
    from tracking.history_pair_runtime import HistoryAssociationTracker
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.summarize_history_tracker_20261028 import summarize
    folder=RUN/'history_tracker_pilot';p=read(folder/'protocol.json');frozen=read(folder/'frozen.json')
    for f in [RUN/'history_two_epoch_pilot',RUN/'history_two_epoch_audit',RUN/'history_tracker_checks']:
        c=read(f/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert read(RUN/'history_two_epoch_audit/report.json')['passed']
    assert read(RUN/'history_tracker_checks/report.json')['passed']
    for file,digest in p['sha256'].items():assert sha(ROOT/file)==digest,file
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    per={name:{} for name in ['native','zero',*frozen['models']]};completed=[]
    def status(**kw):
        if time.time()>end-20:raise TimeoutError('Original history full tracker budget')
        write(folder/'status.json',dict(state='running',updated=now(),**kw),required=False)
    with GpuLock('Local\\DissertationSingleGpu'):
        encoder=build();encoder.eval();buffers={k:v.clone() for k,v in encoder.named_buffers()};models={}
        for name,spec in frozen['models'].items():
            assert sha(ROOT/spec['checkpoint'])==spec['sha256']
            checkpoint=torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=False)
            model=DeterministicHistoryPairHead();model.load_state_dict(checkpoint['state']);models[name]=model.cuda().eval()
        for vi,source in enumerate(p['sources']):
            v=source['video'];out=folder/'videos'/v;out.mkdir(parents=True,exist_ok=True)
            if (out/'report.json').exists():
                e=read(out/'report.json');assert all(sha(ROOT/k)==h for k,h in e['sha256'].items())
                for name,metrics in e['metrics'].items():per[name][v]=metrics
                completed.append(e);continue
            started=time.monotonic();expected=read(ROOT/source['baseline']);align=read(ROOT/source['alignment'])['matches'];assert len(expected)==len(align)
            anchors={int(a['video_frame_index']):i for i,a in enumerate(align)};assert len(anchors)==len(align)
            cache=load_detection_cache(ROOT/source['detections']);capture=cv2.VideoCapture(str(ROOT/source['video_path']));assert capture.isOpened()
            args=SimpleNamespace(**p['tracker_config']);dummy=models['v134_seed42']
            trackers={'native':HistoryAssociationTracker(args,dummy,'0',0),'zero':HistoryAssociationTracker(args,dummy,'134',0)}
            trackers.update({n:HistoryAssociationTracker(args,m,frozen['models'][n]['variant'],p['alpha']) for n,m in models.items()})
            rows={n:[] for n in trackers};raw={n:[] for n in trackers};events={n:[] for n in trackers}
            motion=CausalMotionHistory(**p['motion']);server=FrameMasaServer(encoder)
            roi_frame=[];roi_det=[];embeddings=[];images=[];camera=[];epoch=[];query_frame=[];query_camera=[];query_epoch=[]
            for frame in range(int(cache['frame_count'])):
                ok,image=capture.read();assert ok
                boxes,scores,classes=detections_for_frame(cache,frame);context=None
                if frame in anchors:
                    state=motion.update(image,frame);chosen=np.flatnonzero(scores>=p['tracker_config']['track_low_thresh'])
                    server.start_frame(frame,image);vectors=server.features([server.observation(boxes[d]) for d in chosen]) if len(chosen) else []
                    rois={}
                    for d,vector in zip(chosen,vectors):
                        pixels=motion.crop(boxes[d],image.shape,p['image_size'],p['context'])
                        rois[int(d)]=dict(embedding=vector,image=pixels)
                        roi_frame.append(frame);roi_det.append(int(d));embeddings.append(vector.copy());images.append(pixels)
                        camera.append(state['camera'].copy());epoch.append(state['epoch'])
                    server.finish_frame();query_frame.append(frame);query_camera.append(state['camera'].copy());query_epoch.append(state['epoch'])
                    context=dict(rois=rois,camera=state['camera'],epoch=state['epoch'],video_size=np.array([image.shape[1],image.shape[0]],np.int32))
                for name,tracker in trackers.items():
                    result=tracker.update_frame(frame,boxes,scores,classes,None if name=='native' else context)
                    if name=='native':native_output=result.copy()
                    elif name=='zero':assert np.array_equal(result,native_output),(v,frame,'zero full output differs')
                    if len(result):raw[name].append(np.column_stack([np.full(len(result),frame),result[:,:7]]).astype(np.float32))
                    if frame in anchors:
                        ref=anchors[frame];template={k:expected[ref][k] for k in ['name','frameIndex']}
                        row=scalabel_frame(v,template,result)
                        if name in ['native','zero']:assert row==expected[ref],(name,v,frame)
                        rows[name].append(row)
                    if tracker.events:events[name].extend(tracker.events);tracker.events.clear()
                for attr in ['tracked_stracks','lost_stracks','removed_stracks']:
                    a=getattr(trackers['native'],attr);b=getattr(trackers['zero'],attr)
                    assert len(a)==len(b) and all(x.track_id==y.track_id and np.array_equal(x.mean,y.mean) and np.array_equal(x.covariance,y.covariance) for x,y in zip(a,b)),(v,frame,'zero state differs')
                if frame%20==0:status(stage='Реальные32трекера/назначения/состояния',video=v,index=vi,total=40,frame=frame,frames=int(cache['frame_count']))
            capture.release();hashes={};metrics={}
            feature_folder=out/'visual';feature_folder.mkdir(exist_ok=True)
            em=np.asarray(embeddings,np.float32).reshape(-1,256);im=np.asarray(images,np.uint8).reshape(-1,3,p['image_size'],p['image_size'])
            assert np.isfinite(em).all() and np.allclose(np.linalg.norm(em,axis=1),1,atol=1e-5)
            np.save(feature_folder/'embeddings.npy',em);np.save(feature_folder/'images.npy',im)
            np.savez_compressed(feature_folder/'index.npz',roi_frame=np.asarray(roi_frame,np.int32),roi_det=np.asarray(roi_det,np.int32),camera=np.asarray(camera,np.float32),camera_epoch=np.asarray(epoch,np.int32),
                query_frame=np.asarray(query_frame,np.int32),query_camera=np.asarray(query_camera,np.float32),query_epoch=np.asarray(query_epoch,np.int32),video_size=np.array([image.shape[1],image.shape[0]],np.int32))
            for file in feature_folder.iterdir():hashes[file.relative_to(ROOT).as_posix()]=sha(file)
            for name in trackers:
                values=np.concatenate(raw[name]) if raw[name] else np.zeros((0,8),np.float32)
                if name=='native':
                    with np.load(ROOT/source['raw_array']) as z:assert np.array_equal(values,z[source['raw_array_key']])
                rawfile=out/(name+'.npz');np.savez_compressed(rawfile,outputs=values)
                jsonfile=folder/name/(v+'.json');write(jsonfile,rows[name]);eventfile=out/'events'/(name+'.json');write(eventfile,events[name])
                for file in [rawfile,jsonfile,eventfile]:hashes[file.relative_to(ROOT).as_posix()]=sha(file)
            assert all(torch.equal(buffers[k],value) for k,value in encoder.named_buffers())
            write(out/'tracking_completed.json',dict(video=v,trackers=32,sha256=hashes.copy(),tracking_seconds=time.monotonic()-started,
                source_frozen_sha256=sha(folder/'frozen.json'),native_raw_json_exact=True,zero_raw_json_state_exact=True,
                shadow_pairs=trackers['zero'].shadow_pairs,roi_count=len(roi_frame),batchnorm_unchanged=True,
                note='All GPU inference/raw/JSON/events persisted before metrics; recover CPU metrics without repeating GPU'))
            for name in trackers:
                status(stage='ПолнаяHOTA видео',video=v,index=vi,variant=name)
                metrics[name]=source['baseline_metrics'] if name in ['native','zero'] else evaluate_hota(folder/name,[v],split='val')
                write(out/'metrics'/(name+'.json'),metrics[name])
                per[name][v]=metrics[name]
            assert all(torch.equal(buffers[k],v) for k,v in encoder.named_buffers())
            entry=dict(video=v,metrics=metrics,sha256=hashes,seconds=time.monotonic()-started,native_raw_json_exact=True,zero_raw_json_state_exact=True,
                shadow_pairs=trackers['zero'].shadow_pairs,roi_count=len(roi_frame),batchnorm_unchanged=True,all_query_detections_available=True,ground_truth_in_predictor=False)
            write(out/'report.json',entry);completed.append(entry);write(folder/'completed_videos.json',completed)
            if vi==0:
                need=2*entry['seconds']*(len(p['sources'])-1)+900;available=end-time.time()
                price=dict(first_full_video_seconds=entry['seconds'],conservative_remaining_seconds=need,available_seconds=available,proceed=need<available);write(folder/'price.json',price)
                if not price['proceed']:
                    write(folder/'report.json',dict(complete=False,stopped_on_cost=True,completed=completed,remaining=p['sources'][1:],scientific_verdict=None,updated=now()));return
        pooled={}
        for name in trackers:
            status(stage='Совокупные метрики40видео',variant=name)
            pooled[name]=pooled['native'] if name=='zero' else evaluate_hota(folder/name,[s['video'] for s in p['sources']],split='val')
        result=summarize(per,pooled,p)
        result.update(complete=True,completed=completed,per_video=per,pooled=pooled,frozen_sha256=sha(folder/'frozen.json'),ground_truth_in_predictor=False,batchnorm_unchanged=True,epochs=2,updated=now())
        write(folder/'report.json',result);write(folder/'status.json',dict(state='completed',updated=now()))


if __name__=='__main__':main()
