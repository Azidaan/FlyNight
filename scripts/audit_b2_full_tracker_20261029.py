"""CPUповтор всей Б2: все входы/головы, записанные GPUдетекции для точных ID."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_tracker_audit';source=RUN/'b2_tracker_pilot'
    p=service.read(source/'protocol.json');freeze=service.read(source/'frozen.json')
    c=service.read(source/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    report=service.read(source/'report.json')
    if not report['complete']:
        service.write(folder/'report.json',dict(passed=False,full_series=False,scientific_verdict=None,reason='Full40notcomplete;notnegative science',updated=service.now()));return
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    import cv2,numpy as np,torch
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.set_default_device('cpu')
    from ultralytics import YOLO
    from tracking.b2_pilot_runtime import load_branches,modified_features,IndependentByteTracker,assert_tracker_state_equal
    from tracking.b2_feature_temporal import detector_features,detection_training_predictions
    from tracking.b2_causal_detector import FeatureRing
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from scripts.summarize_b2_tracker_20261029 import summarize
    from evaluation.hota_eval import evaluate_hota
    assert service.sha(ROOT/freeze['checkpoint'])==freeze['checkpoint_sha256']
    models=load_branches(torch.load(ROOT/freeze['checkpoint'],map_location='cpu',weights_only=False),'cpu')
    yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
    cap=cv2.VideoCapture(str(ROOT/p['sources'][0]['video_path']));ok,image=cap.read();cap.release();assert ok
    yolo.predict(image,imgsz=960,conf=.01,iou=.7,device='cpu',verbose=False)
    predictor=yolo.predictor;detector=predictor.model.model;detector.eval().float().requires_grad_(False)
    head=detector.model[-1]
    end=min(datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp(),datetime.fromisoformat(p['audit_absolute_end']).timestamp())
    names=['native','zero','C','D','R1','B','last','mean'];completed=[];maximum=0.;checked=0
    for vi,spec in enumerate(p['sources']):
        v=spec['video'];out=source/'videos'/v;entry=service.read(out/'report.json')
        assert entry['video']==v and all(service.sha(ROOT/f)==h for f,h in entry['sha256'].items())
        key=dict(video_report_sha256=service.sha(out/'report.json'),frozen_sha256=service.sha(source/'frozen.json'),protocol_sha256=service.sha(source/'protocol.json'))
        marker=folder/'completed'/(v+'.json')
        if marker.exists():
            old=service.read(marker);assert old['inputs']==key
            completed.append(old);maximum=max(maximum,old['max_raw_head_difference']);checked+=old['head_arrays_checked'];continue
        began=time.monotonic();cache=load_detection_cache(ROOT/spec['detections']);expected=service.read(ROOT/spec['baseline'])
        anchors={int(a['video_frame_index']):i for i,a in enumerate(service.read(ROOT/spec['alignment'])['matches'])}
        trackers={n:IndependentByteTracker(SimpleNamespace(**p['tracker_config'])) for n in names}
        raw={n:[] for n in names};rows={n:[] for n in names};ring=FeatureRing(5)
        capture=cv2.VideoCapture(str(ROOT/spec['video_path']));fps=capture.get(cv2.CAP_PROP_FPS)
        local_max=0.;local_checked=0
        for frame in range(int(cache['frame_count'])):
            if time.time()>end-30:raise TimeoutError('B2fullCPUaudit originaldeadline')
            ok,image=capture.read();assert ok
            original=detections_for_frame(cache,frame);dets={n:original for n in names}
            if frame in anchors:
                tensor=predictor.preprocess([image]);features=detector_features(detector,tensor)
                ring.append(frame,features);sequence,frames=ring.sequence();active=len(frames)==5
                with np.load(out/'heads'/f'{frame:07d}.npz') as values: saved={k:values[k] for k in values.files}
                with np.load(out/'detections'/f'{frame:07d}.npz') as values: recorded={k:values[k] for k in values.files}
                assert np.array_equal(saved['history_frames'],frames) and bool(saved['active'])==active and float(saved['fps'])==fps
                assert np.array_equal(saved['input_shape'],tensor.shape[-2:])
                with torch.no_grad():
                    predictions={'native':detection_training_predictions(head,features)}
                    if active:
                        for name,model in models.items():
                            modified,_=modified_features(model,sequence,frames,fps,name)
                            predictions[name]=detection_training_predictions(head,[modified,*features[1:]])
                    for name,pred in predictions.items():
                        for field in ('boxes','scores'):
                            current=pred[field].cpu().numpy();reference=saved[name+'_'+field]
                            assert current.shape==reference.shape and np.isfinite(current).all()
                            diff=float(np.abs(current-reference).max());local_max=max(local_max,diff);local_checked+=1
                            assert diff<=p['raw_head_cpu_gpu_tolerance'],(v,frame,name,field,diff)
                        # The full neural head was recomputed above. Decode the
                        # saved GPUraw after that check to verify postprocessing
                        # without allowing floating ties to change discrete IDs.
                        rawpred=dict(boxes=torch.from_numpy(saved[name+'_boxes']),scores=torch.from_numpy(saved[name+'_scores']),feats=features)
                        decoded=head._inference(rawpred).permute(0,2,1)[0]
                        ids=torch.from_numpy(saved[name+'_selected_anchor']).long()
                        cls=torch.from_numpy(saved[name+'_selected_class']).long()
                        score=decoded[ids,4+cls]
                        selected=torch.cat((decoded[ids,:4],score[:,None],cls[:,None].float()),1).numpy()
                        gpu=saved[name+'_decoded'][0]
                        assert np.max(np.abs(selected[:,:4]-gpu[:,:4]))<=p['postprocess_box_tolerance_pixels']
                        assert np.max(np.abs(selected[:,4]-gpu[:,4]))<=p['postprocess_score_tolerance']
                        assert np.array_equal(selected[:,5],gpu[:,5])
                        outside=decoded[:,4:].clone();outside[ids,cls]=-1
                        assert outside.max()<=score.min()+p['postprocess_score_tolerance']
                        result=predictor.postprocess((torch.from_numpy(saved[name+'_decoded'].copy()),rawpred),tensor,[image])[0].boxes
                        boxes=recorded[name+'_xyxy'];scores=recorded[name+'_conf'];classes=recorded[name+'_cls']
                        assert result.xyxy.shape==boxes.shape
                        if len(boxes):assert np.max(np.abs(result.xyxy.numpy()-boxes))<=p['postprocess_box_tolerance_pixels']
                        assert np.array_equal(result.conf.numpy(),scores) and np.array_equal(result.cls.numpy().astype(np.int16),classes)
                        if name=='native':assert all(np.array_equal(a,b) for a,b in zip((boxes,scores,classes),original))
                        else:dets[name]=(boxes,scores,classes)
            for name,tracker in trackers.items():
                output=tracker.step(*dets[name])
                if name=='native':native=output.copy()
                elif name=='zero':assert np.array_equal(native,output)
                if len(output):raw[name].append(np.column_stack([np.full(len(output),frame),output[:,:7]]).astype(np.float32))
                if frame in anchors:
                    ref=anchors[frame];row=scalabel_frame(v,{k:expected[ref][k] for k in ('name','frameIndex')},output)
                    if name in ('native','zero'):assert row==expected[ref]
                    rows[name].append(row)
            assert_tracker_state_equal(trackers['native'],trackers['zero'])
            if frame%50==0:service.write(folder/'status.json',dict(stage='Full CPUhead/raw/tracker replay',video=v,index=vi,total=40,frame=frame,max_raw_head_difference=max(maximum,local_max),updated=service.now()))
        capture.release()
        for name in names:
            actual=np.concatenate(raw[name]) if raw[name] else np.zeros((0,8),np.float32)
            with np.load(out/(name+'.npz')) as z:assert np.array_equal(actual,z['outputs'])
            assert rows[name]==service.read(source/name/(v+'.json'))
            metric=evaluate_hota(source/name,[v],split='val')
            assert metric==report['per_video'][name][v],(v,name,'metrics differ')
        item=dict(video=v,inputs=key,passed=True,max_raw_head_difference=local_max,head_arrays_checked=local_checked,seconds=time.monotonic()-began)
        service.write(marker,item);completed.append(item);maximum=max(maximum,local_max);checked+=local_checked
        if vi==0:
            need=2*item['seconds']*39+300;available=end-time.time()
            service.write(folder/'price.json',dict(proceed=need<available,first_complete_video_seconds=item['seconds'],conservative_remaining_seconds=need,available_seconds=available))
            if need>=available:
                service.write(folder/'report.json',dict(passed=False,full_series=False,stopped_on_cost=True,completed=completed,scientific_verdict=None,updated=service.now()));return
    pooled={n:evaluate_hota(source/n,[s['video'] for s in p['sources']],split='val') for n in names}
    assert pooled==report['pooled']
    result=summarize(report['per_video'],pooled,p['sources'])
    assert all(report[k]==v for k,v in result.items()) and not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(passed=True,full_series=True,videos=40,trackers=8,completed=completed,
                  raw_head_arrays_checked=checked,max_raw_head_cpu_gpu_difference=maximum,head_tolerance=p['raw_head_cpu_gpu_tolerance'],
                  postprocess_checked=True,discrete_tracker_replay_uses_validated_recorded_gpu_detections=True,
                  raw_json_zero_states_metrics_exact=True,paired_recomputed=True,main_acceptance=result['main_acceptance'],updated=service.now()))
    service.write(source/'independent_audit.json',service.read(folder/'report.json'))


if __name__=='__main__':main()
