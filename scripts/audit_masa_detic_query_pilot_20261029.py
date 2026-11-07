"""CPU replay of every assignment/raw/JSON/metric; declared feature spot checks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from types import SimpleNamespace
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np,torch,cv2
    from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    folder=RUN/'masa_query_audit';gpu=RUN/'masa_query_pilot';p=service.read(gpu/'protocol.json')
    c=service.read(gpu/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    report=service.read(gpu/'report.json');assert report['complete'] and len(report['completed'])==20
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    torch.set_num_threads(2);torch.set_default_device('cpu');cv2.setNumThreads(1)
    class Replay(FrameMasaServer):
        def features(self,tokens,null_seed=None):
            for token in tokens:
                if token.vector is None:
                    assert token.frame==self.frame
                    token.vector=self.values[tuple(token.crop.tolist())]
            return np.stack([t.vector for t in tokens])
    checked=[];per={n:{} for n in p['variants']};feature_checks=[]
    for vi,source in enumerate(p['sources']):
        video=source['video'];out=gpu/'videos'/video;marker=folder/'completed'/(video+'.json')
        item=report['completed'][vi];assert item['video']==video
        for f,h in item['sha256'].items():assert service.sha(ROOT/f)==h,f
        marker_key=dict(video_report_sha=service.sha(out/'report.json'),protocol_sha=service.sha(gpu/'protocol.json'))
        if marker.exists():
            old=service.read(marker);assert old['inputs']==marker_key
            checked.append(old)
            for n,m in old['metrics'].items():per[n][video]=m
            continue
        cache=load_detection_cache(ROOT/source['detections']);baseline=service.read(ROOT/source['baseline'])
        align=service.read(ROOT/source['alignment'])['matches'];anchors={int(x['video_frame_index']):i for i,x in enumerate(align)}
        vectors=np.load(out/'vectors.npz');expected=np.load(out/'raw.npz')
        servers={n:Replay(None) for n in ['r50_1024','r50_1333','detic_1333']}
        args=SimpleNamespace(**p['config']);weight=p['appearance_weight']
        trackers={'native':QueryAppearanceTracker(args),'zero':QueryAppearanceTracker(args,servers['detic_1333'],weight,True)}
        trackers.update({n:QueryAppearanceTracker(args,s,weight) for n,s in servers.items()})
        rows={n:[] for n in trackers};raw={n:[] for n in trackers}
        for frame in range(int(cache['frame_count'])):
            if frame%50==0:service.write(folder/'status.json',dict(stage='Exact CPU replay',video=video,index=vi,total=20,frame=frame,updated=service.now()))
            active=frame in anchors
            if active:
                for name,server in servers.items():
                    server.start_frame(frame,None)
                    boxes=vectors[f'{frame}_{name}_boxes'];z=vectors[f'{frame}_{name}_vectors']
                    assert z.shape==(len(boxes),256) and np.isfinite(z).all()
                    server.values={tuple(b.tolist()):v for b,v in zip(boxes,z)}
            boxes,scores,classes=detections_for_frame(cache,frame);detections=_DetSlice(boxes,scores,classes)
            outputs={n:t.update_query_frame(detections,active) for n,t in trackers.items()}
            assert np.array_equal(outputs['native'],outputs['zero']);assert_states_equal(trackers['native'],trackers['zero'])
            if active:
                for server in servers.values():server.finish_frame()
            for name,value in outputs.items():
                if len(value):raw[name].append(np.column_stack([np.full(len(value),frame),value[:,:7]]).astype(np.float32))
                if active:
                    index=anchors[frame];row=scalabel_frame(video,{k:baseline[index][k] for k in ['name','frameIndex']},value);rows[name].append(row)
        metrics={}
        for name in trackers:
            actual=np.concatenate(raw[name]) if raw[name] else np.zeros((0,8),np.float32)
            assert np.array_equal(actual,expected[name]),(video,name,'raw')
            assert rows[name]==service.read(out/(name+'.json')),(video,name,'JSON')
            assert trackers[name].events==service.read(out/(name+'_events.json')),(video,name,'events')
            dest=folder/'metric_inputs'/name;service.write(dest/(video+'.json'),rows[name])
            metrics[name]=evaluate_hota(dest,[video],split='val')
            assert metrics[name]==item['metrics'][name],(video,name,'metrics')
            per[name][video]=metrics[name]
        proof=dict(video=video,inputs=marker_key,passed=True,metrics=metrics,raw_json_events_exact=True)
        service.write(marker,proof);checked.append(proof)
    # Feature spot checks are prospectively restricted to query frames in videos
    # 0/10/19. Every discrete tracker output above is replayed, not sampled.
    from tracking.masa_r50 import build as r50_build,preprocess as input1024
    from tracking.masa_detic_prepared import build as detic_build
    from tracking.masa_detic_bdd_input import preprocess_bdd
    models={'r50':r50_build('cpu'),'detic':detic_build('cpu')}
    for vi in p['feature_audit_video_indices']:
        source=p['sources'][vi];video=source['video'];out=gpu/'videos'/video
        ledger=np.load(out/'vectors.npz');frame=service.read(out/'tracking_completed.json')['query_frames'][0]
        capture=cv2.VideoCapture(str(ROOT/source['video_path']))
        try:
            for _ in range(frame+1):ok,image=capture.read();assert ok
        finally:capture.release()
        for name in ['r50_1024','r50_1333','detic_1333']:
            boxes=ledger[f'{frame}_{name}_boxes'];ref=ledger[f'{frame}_{name}_vectors']
            if not len(boxes):continue
            model=models['detic' if name.startswith('detic') else 'r50']
            tensor,scale=(input1024 if name.endswith('1024') else preprocess_bdd)(image,'cpu')
            with torch.inference_mode():
                actual=torch.nn.functional.normalize(model.embed_boxes(model.features(tensor),torch.as_tensor(boxes),scale),dim=1).numpy()
            difference=float(np.max(np.abs(actual-ref)))
            assert difference<=p['feature_cpu_gpu_normalized_tolerance'],(video,name,difference)
            feature_checks.append(dict(video=video,frame=frame,variant=name,rois=len(boxes),max_difference=difference))
    comparisons={}
    for control in ['native','r50_1333','r50_1024']:
        comparisons[control]=paired({v:m['HOTA'] for v,m in per['detic_1333'].items()}, {v:m['HOTA'] for v,m in per[control].items()},n_boot=20000)
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,complete=True,videos=20,trackers=5,checked=checked,feature_checks=feature_checks,paired= comparisons,
                main_acceptance=all(comparisons[k]['ci95'][0]>0 for k in ['native','r50_1333']),
                scientific_scope='Frozen normalized features and query-schedule association in our tracker; not full official MASA tracker or FlyVis core',updated=service.now())
    service.write(folder/'report.json',result);service.write(gpu/'independent_audit.json',result)

if __name__=='__main__':main()
