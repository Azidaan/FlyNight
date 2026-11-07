"""Frozen appearance extractors, matched association, full native 30Hz tracker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys,time,faulthandler,shutil
from datetime import datetime
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import cv2,numpy as np,torch
    from tracking.masa_r50 import build as build_r50,preprocess as input1024
    from tracking.masa_detic_prepared import build as build_detic
    from tracking.masa_detic_bdd_input import preprocess_bdd
    from tracking.query_appearance_comparison import QueryAppearanceTracker,FrozenFrameServer,assert_states_equal
    from tracking.offline_association import _DetSlice
    from tracking.gpu_lock import GpuLock
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    folder=RUN/'masa_query_fp32_recovery1';p=service.read(folder/'protocol.json')
    c=service.read(RUN/'masa_output_interface/controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and service.read(RUN/'masa_output_interface/report.json')['passed']
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    deadline=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp()
    def status(**kw):
        assert shutil.disk_usage(ROOT).free>5*1024**3,'Disk5GBreserve'
        assert time.time()<deadline-30,'Global save deadline'
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(600,exit=True)
        service.write(folder/'status.json',dict(updated=service.now(),**kw))
    with GpuLock('Local\\DissertationSingleGpu'):
        status(stage='Loading frozen original checkpoints')
        r50=build_r50('cuda');detic=build_detic('cuda')
        r50.requires_grad_(False);detic.requires_grad_(False)
        buffers=[{k:v.clone() for k,v in m.named_buffers()} for m in [r50,detic]]
        names=p['variants'];completed=[]
        for vi,source in enumerate(p['sources']):
            video=source['video'];out=folder/'videos'/video;out.mkdir(parents=True,exist_ok=True)
            if (out/'report.json').exists():
                done=service.read(out/'report.json')
                assert all(service.sha(ROOT/f)==h for f,h in done['sha256'].items())
                completed.append(done);continue
            assert not (out/'tracking_completed.json').exists(),'Saved GPU exists; recover only CPU metrics'
            began=time.monotonic();cache=load_detection_cache(ROOT/source['detections'])
            baseline=service.read(ROOT/source['baseline'])
            alignment=service.read(ROOT/source['alignment'])['matches']
            anchors={int(x['video_frame_index']):i for i,x in enumerate(alignment)}
            assert len(anchors)==len(alignment)==len(baseline)
            servers={'r50_1024':FrozenFrameServer(r50,input1024),'r50_1333':FrozenFrameServer(r50,preprocess_bdd),
                     'detic_1333':FrozenFrameServer(detic,preprocess_bdd)}
            args=SimpleNamespace(**p['config']);weight=p['appearance_weight']
            trackers={'native':QueryAppearanceTracker(args),
                      'zero':QueryAppearanceTracker(args,servers['detic_1333'],weight,True)}
            trackers.update({name:QueryAppearanceTracker(args,server,weight) for name,server in servers.items()})
            rows={n:[] for n in names};raw={n:[] for n in names};ledger={}
            capture=cv2.VideoCapture(str(ROOT/source['video_path']))
            for frame in range(int(cache['frame_count'])):
                ok,image=capture.read();assert ok
                active=frame in anchors
                if frame%20==0:status(stage='Full tracker on immutable detections',video=video,index=vi,total=20,frame=frame,total_frames=int(cache['frame_count']))
                if active:
                    for server in servers.values():server.start_frame(frame,image)
                boxes,scores,classes=detections_for_frame(cache,frame);detections=_DetSlice(boxes,scores,classes)
                outputs={n:t.update_query_frame(detections,active) for n,t in trackers.items()}
                assert np.array_equal(outputs['native'],outputs['zero'])
                assert_states_equal(trackers['native'],trackers['zero'])
                if active:
                    for name,server in servers.items():
                        server.finish_frame()
                        keys=list(server.current_ledger)
                        ledger[f'{frame}_{name}_boxes']=np.array([k[1:] for k in keys],np.float32).reshape(-1,4)
                        ledger[f'{frame}_{name}_vectors']=np.array([server.current_ledger[k] for k in keys],np.float32).reshape(-1,256)
                for name,value in outputs.items():
                    if len(value):raw[name].append(np.column_stack([np.full(len(value),frame),value[:,:7]]).astype(np.float32))
                    if active:
                        index=anchors[frame];template={k:baseline[index][k] for k in ['name','frameIndex']}
                        row=scalabel_frame(video,template,value)
                        if name in ['native','zero']:assert row==baseline[index],(video,index,name)
                        rows[name].append(row)
            capture.release()
            arrays={n:np.concatenate(v) if v else np.zeros((0,8),np.float32) for n,v in raw.items()}
            native=np.load(ROOT/source['raw_array'])[source['raw_array_key']]
            assert np.array_equal(arrays['native'],native)
            np.savez_compressed(out/'raw.npz',**arrays);np.savez_compressed(out/'vectors.npz',**ledger)
            for name in names:
                service.write(out/(name+'.json'),rows[name]);service.write(out/(name+'_events.json'),trackers[name].events)
            hashes={f.relative_to(ROOT).as_posix():service.sha(f) for f in out.iterdir() if f.is_file()}
            service.write(out/'tracking_completed.json',dict(sha256=hashes,video=video,frames=int(cache['frame_count']),query_frames=sorted(anchors),native_zero_exact=True,updated=service.now()))
            metrics={}
            for name in names:
                status(stage='Full HOTA per video',video=video,index=vi,variant=name)
                destination=folder/'metric_inputs'/name
                service.write(destination/(video+'.json'),rows[name])
                metrics[name]=source['baseline_metrics'] if name=='native' else evaluate_hota(destination,[video],split='val')
            assert all(all(torch.equal(before[k],v) for k,v in model.named_buffers()) for model,before in zip([r50,detic],buffers))
            item=dict(video=video,metrics=metrics,seconds=time.monotonic()-began,sha256={**hashes,
                (out/'tracking_completed.json').relative_to(ROOT).as_posix():service.sha(out/'tracking_completed.json')},native_zero_exact=True,
                zero_shadow_pairs=trackers['zero'].shadow_pairs,encoded_frames={k:v.encoded_frames for k,v in servers.items()})
            service.write(out/'report.json',item);completed.append(item)
            service.write(folder/'progress.json',dict(completed=completed,scientific_verdict=None))
            if vi==0:
                saved_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
                free=shutil.disk_usage(ROOT).free
                need=2*saved_bytes*19+5*1024**3
                service.write(folder/'disk_gate.json',dict(passed=need<free,required_bytes=need,free_bytes=free,updated=service.now()))
                assert need<free,'Disk gate: preserve first result and stop'
                service.write(folder/'price.json',dict(first_complete_video_seconds=item['seconds'],conservative_remaining_seconds=2*item['seconds']*19+300,
                    available_seconds=deadline-time.time(),time_forecast_advisory=True,stage_time_budget_cancelled=True))
        service.write(folder/'report.json',dict(complete=True,videos=20,variants=names,completed=completed,scientific_verdict=None,audit_pending=True,updated=service.now()))
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
