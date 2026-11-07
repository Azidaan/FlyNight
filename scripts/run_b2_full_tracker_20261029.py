"""Настоящие обнаружения обученной Б2 перед восемью независимыми трекерами."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys,time,shutil,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_tracker_pilot';p=service.read(folder/'protocol.json');freeze=service.read(folder/'frozen.json')
    for name in ('b2_sparse_training_pilot','b2_sparse_training_record','b2_sparse_training_audit','b2_tracker_preparation'):
        c=service.read(RUN/name/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert service.read(RUN/'b2_sparse_training_audit/report.json')['passed']
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from tracking.b2_pilot_runtime import load_branches,modified_features,IndependentByteTracker,assert_tracker_state_equal
    from tracking.b2_feature_temporal import detector_features,detection_training_predictions
    from tracking.b2_causal_detector import FeatureRing
    from tracking.gpu_lock import GpuLock
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from scripts.summarize_b2_tracker_20261029 import summarize
    from evaluation.hota_eval import evaluate_hota
    torch.set_num_threads(2);cv2.setNumThreads(1)
    end=min(datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp(),datetime.fromisoformat(p['gpu_absolute_end']).timestamp())
    names=['native','zero','C','D','R1','B','last','mean'];per={n:{} for n in names};completed=[]
    def status(**kw):
        if time.time()>end-30:raise TimeoutError('B2 tracker original GPUdeadline')
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(300,exit=True)
        service.write(folder/'status.json',dict(updated=service.now(),**kw))
    with GpuLock('Local\\DissertationSingleGpu'):
        assert service.sha(ROOT/freeze['checkpoint'])==freeze['checkpoint_sha256']
        ck=torch.load(ROOT/freeze['checkpoint'],map_location='cpu',weights_only=False)
        models=load_branches(ck,'cuda')
        yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
        cap=cv2.VideoCapture(str(ROOT/p['sources'][0]['video_path']));ok,image=cap.read();cap.release();assert ok
        for _ in range(2):yolo.predict(image,imgsz=960,conf=.01,iou=.7,device='0',verbose=False)
        predictor=yolo.predictor;detector=predictor.model.model
        detector.eval().float().requires_grad_(False);head=detector.model[-1]
        assert head.end2end and not head.agnostic_nms and head.max_det==300
        before={k:v.detach().clone() for k,v in detector.named_parameters()}
        def model_parameters():
            result={}
            for name,(branch,bb) in models.items():
                result.update({(name,'branch',k):v for k,v in branch.named_parameters()})
                if name in ('C','D'):result.update({(name,'core',k):v for k,v in bb.net.named_parameters()})
                elif hasattr(bb,'named_parameters'):result.update({(name,'ordinary',k):v for k,v in bb.named_parameters()})
            return result
        initial_weights={k:v.detach().clone() for k,v in model_parameters().items()}
        for vi,source in enumerate(p['sources']):
            v=source['video'];out=folder/'videos'/v;out.mkdir(parents=True,exist_ok=True)
            if (out/'report.json').exists():
                item=service.read(out/'report.json');assert all(service.sha(ROOT/f)==h for f,h in item['sha256'].items())
                for n,m in item['metrics'].items():per[n][v]=m
                completed.append(item);continue
            assert not(out/'tracking_completed.json').exists(),'GPUalreadycomplete;recoverCPUmetricswithoutrepeating'
            began=time.monotonic();cache=load_detection_cache(ROOT/source['detections'])
            expected=service.read(ROOT/source['baseline']);align=service.read(ROOT/source['alignment'])['matches']
            anchors={int(x['video_frame_index']):i for i,x in enumerate(align)};assert len(anchors)==len(expected)
            capture=cv2.VideoCapture(str(ROOT/source['video_path']));assert capture.isOpened();fps=capture.get(cv2.CAP_PROP_FPS)
            ring=FeatureRing(5);trackers={n:IndependentByteTracker(SimpleNamespace(**p['tracker_config'])) for n in names}
            raw={n:[] for n in names};rows={n:[] for n in names};queries=[];shadow=0
            hashes={}
            for frame in range(int(cache['frame_count'])):
                ok,image=capture.read();assert ok
                native_boxes,native_scores,native_cls=detections_for_frame(cache,frame)
                detections={n:(native_boxes,native_scores,native_cls) for n in names}
                if frame in anchors:
                    tensor=predictor.preprocess([image]);features=detector_features(detector,tensor)
                    ring.append(frame,features);sequence,frames=ring.sequence();active=len(frames)==5
                    tensors={};ledger={};detector_arrays={}
                    with torch.no_grad():
                        baseline=detection_training_predictions(head,features)
                        tensors['native']=baseline
                        if active:
                            for name,model in models.items():
                                modified,_=modified_features(model,sequence,frames,fps,name)
                                tensors[name]=detection_training_predictions(head,[modified,*features[1:]])
                            # D was really computed above. Zero discards its
                            # correction BEFORE the original detection head.
                            zero=detection_training_predictions(head,[sequence[-1],*features[1:]])
                            assert all(torch.equal(zero[k],baseline[k]) for k in ('boxes','scores'));shadow+=1
                        for name,pred in tensors.items():
                            assert all(torch.isfinite(pred[k]).all() for k in ('boxes','scores'))
                            ledger[name+'_boxes']=pred['boxes'].cpu().numpy()
                            ledger[name+'_scores']=pred['scores'].cpu().numpy()
                            decoded_all=head._inference(pred).permute(0,2,1)
                            top_scores,top_classes,top_indices=head.get_topk_index(decoded_all[:,:,4:],head.max_det)
                            decoded=head.postprocess(decoded_all)
                            ledger[name+'_selected_anchor']=top_indices.cpu().numpy().reshape(-1).astype(np.int32)
                            ledger[name+'_selected_class']=top_classes.cpu().numpy().reshape(-1).astype(np.int16)
                            ledger[name+'_decoded']=decoded.cpu().numpy().copy()
                            result=predictor.postprocess((decoded,pred),tensor,[image])[0].boxes
                            boxes=result.xyxy.cpu().numpy().astype(np.float32);scores=result.conf.cpu().numpy().astype(np.float32);cls=result.cls.cpu().numpy().astype(np.int16)
                            detector_arrays[name+'_xyxy']=boxes;detector_arrays[name+'_conf']=scores;detector_arrays[name+'_cls']=cls
                            if name=='native':
                                assert np.array_equal(boxes,native_boxes) and np.array_equal(scores,native_scores) and np.array_equal(cls,native_cls),(v,frame,'native cache')
                            else:detections[name]=(boxes,scores,cls)
                    ledger['history_frames']=np.asarray(frames,np.int32)
                    ledger['active']=np.asarray(active);ledger['fps']=np.asarray(fps,np.float64)
                    ledger['input_shape']=np.asarray(tensor.shape[-2:],np.int32)
                    file=out/'heads'/f'{frame:07d}.npz';file.parent.mkdir(exist_ok=True);np.savez(file,**ledger)
                    detfile=out/'detections'/f'{frame:07d}.npz';detfile.parent.mkdir(exist_ok=True);np.savez(detfile,**detector_arrays)
                    for f in (file,detfile):hashes[f.relative_to(ROOT).as_posix()]=service.sha(f)
                    queries.append(frame)
                for name,tracker in trackers.items():
                    output=tracker.step(*detections[name])
                    if name=='native':native_output=output.copy()
                    elif name=='zero':assert np.array_equal(output,native_output)
                    if len(output):raw[name].append(np.column_stack([np.full(len(output),frame),output[:,:7]]).astype(np.float32))
                    if frame in anchors:
                        ref=anchors[frame];template={k:expected[ref][k] for k in ('name','frameIndex')}
                        row=scalabel_frame(v,template,output)
                        if name in ('native','zero'):assert row==expected[ref]
                        rows[name].append(row)
                assert_tracker_state_equal(trackers['native'],trackers['zero'])
                if frame%20==0:status(stage='Eight real detector/trackers',video=v,index=vi,total=40,frame=frame,frames=int(cache['frame_count']))
            capture.release()
            for name in names:
                values=np.concatenate(raw[name]) if raw[name] else np.zeros((0,8),np.float32)
                if name=='native':
                    with np.load(ROOT/source['raw_array']) as z:assert np.array_equal(values,z[source['raw_array_key']])
                file=out/(name+'.npz');np.savez_compressed(file,outputs=values)
                jsonfile=folder/name/(v+'.json');service.write(jsonfile,rows[name])
                for f in (file,jsonfile):hashes[f.relative_to(ROOT).as_posix()]=service.sha(f)
            assert all(torch.equal(before[k],value) for k,value in detector.named_parameters())
            assert all(torch.equal(initial_weights[k],value) for k,value in model_parameters().items())
            tracking=dict(video=v,sha256=hashes,queries=queries,shadow_queries=shadow,native_zero_raw_json_state_exact=True,
                          training_frozen_sha256=service.sha(folder/'frozen.json'),tracking_seconds=time.monotonic()-began,
                          raw_head_bytes=sum((ROOT/f).stat().st_size for f in hashes if '/heads/' in f),ground_truth_in_predictor=False,all_model_weights_unchanged=True)
            service.write(out/'tracking_completed.json',tracking)
            metrics={}
            for name in names:
                status(stage='Full HOTA',video=v,index=vi,variant=name)
                metrics[name]=source['baseline_metrics'] if name in ('native','zero') else evaluate_hota(folder/name,[v],split='val')
                service.write(out/'metrics'/(name+'.json'),metrics[name]);per[name][v]=metrics[name]
            item={**tracking,'metrics':metrics,'seconds':time.monotonic()-began}
            service.write(out/'report.json',item);completed.append(item)
            if vi==0:
                required=2*item['seconds']*39+600;available=end-time.time()
                disk_need=2*sum((ROOT/f).stat().st_size for f in hashes)*39+5_000_000_000
                free=shutil.disk_usage(folder).free
                price=dict(proceed=required<available and disk_need<free,first_full_video_seconds=item['seconds'],
                           conservative_remaining_seconds=required,available_seconds=available,conservative_remaining_disk_bytes=disk_need,available_disk_bytes=free)
                service.write(folder/'price.json',price)
                if not price['proceed']:
                    service.write(folder/'report.json',dict(complete=False,completed=completed,stopped_on_cost=True,scientific_verdict=None,updated=service.now()));return
        pooled={}
        for name in names:
            status(stage='Pooled40 HOTA',variant=name)
            pooled[name]=pooled['native'] if name=='zero' else evaluate_hota(folder/name,[x['video'] for x in p['sources']],split='val')
        result=summarize(per,pooled,p['sources'])
        result.update(complete=True,completed=completed,per_video=per,pooled=pooled,updated=service.now())
        service.write(folder/'report.json',result)
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
