"""Prospective full-video development check, then separate CPU replay."""
import os
from pathlib import Path
import sys,time,faulthandler,shutil
from datetime import datetime
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    audit='--audit' in sys.argv
    os.environ['CUDA_VISIBLE_DEVICES']='' if audit else '0'
    import cv2,numpy as np,torch
    from tracking.query_memory_write import MemoryWriteTracker
    from tracking.query_appearance_comparison import FrozenFrameServer,assert_states_equal
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    gpu=RUN/'memory_write_dev';folder=RUN/('memory_write_dev_audit' if audit else 'memory_write_dev')
    p=service.read(gpu/'protocol.json');source=p['source'];video=source['video']
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    assert service.read(RUN/'memory_write_preparation/controller_result.json')['exit_code']==0
    if audit:
        c=service.read(gpu/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        marker=service.read(gpu/'tracking_completed.json')
        assert all(service.sha(ROOT/f)==h for f,h in marker['sha256'].items())
    else:
        assert not (gpu/'tracking_completed.json').exists(),'Completed GPU exists: replay/metrics only'
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not audit:
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    else:torch.set_default_device('cpu')
    deadline=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp()
    def status(**kw):
        assert time.time()<deadline-30,'Global save deadline'
        faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(600,exit=True)
        if not audit:assert shutil.disk_usage(ROOT).free>5_000_000_000,'Disk reserve5GB'
        service.write(folder/'status.json',dict(updated=service.now(),**kw))
    class Replay(FrameMasaServer):
        def features(self,tokens,null_seed=None):
            for token in tokens:
                if token.vector is None:
                    assert token.frame==self.frame
                    token.vector=self.values[tuple(token.crop.tolist())].copy()
            return np.stack([t.vector for t in tokens])
    lock=None;models=None
    if audit:
        ledger=np.load(gpu/'vectors.npz');expected=np.load(gpu/'raw.npz')
        servers={n:Replay(None) for n in ['r50','detic']}
    else:
        from tracking.gpu_lock import GpuLock
        from tracking.masa_r50 import build as r50_build
        from tracking.masa_detic_prepared import build as detic_build
        from tracking.masa_detic_bdd_input import preprocess_bdd
        lock=GpuLock('Local\\DissertationSingleGpu');lock.__enter__()
        status(stage='Loading immutable FP32 feature extractors')
        models={'r50':r50_build('cuda'),'detic':detic_build('cuda')}
        for m in models.values():m.requires_grad_(False)
        frozen={n:{k:v.clone() for k,v in m.named_buffers()} for n,m in models.items()}
        servers={n:FrozenFrameServer(m,preprocess_bdd) for n,m in models.items()};ledger={}
    args=SimpleNamespace(**p['config']);weight=p['appearance_weight']
    trackers={'native':MemoryWriteTracker(args),
        'zero':MemoryWriteTracker(args,servers['detic'],weight,True,'cautious')}
    for encoder in servers:
        for policy in ['last','mean','cautious']:
            trackers[encoder+'_'+policy]=MemoryWriteTracker(args,servers[encoder],weight,False,policy,p['beta'])
    cache=load_detection_cache(ROOT/source['detections'])
    align=service.read(ROOT/source['alignment'])['matches'];anchors={int(x['video_frame_index']):x for x in align}
    assert len(anchors)==len(align)
    capture=None if audit else cv2.VideoCapture(str(ROOT/source['video_path']))
    rows={n:[] for n in trackers};raw={n:[] for n in trackers};began=time.monotonic()
    for frame in range(int(cache['frame_count'])):
        if frame%20==0:status(stage='Exact CPU replay' if audit else 'Full development tracker',video=video,frame=frame,total_frames=int(cache['frame_count']))
        image=None
        if capture is not None:
            ok,image=capture.read();assert ok
        active=frame in anchors
        boxes,scores,classes=detections_for_frame(cache,frame)
        if active:
            for name,s in servers.items():
                s.start_frame(frame,image)
                if audit:s.values={tuple(b.tolist()):v for b,v in zip(ledger[f'{frame}_{name}_boxes'],ledger[f'{frame}_{name}_vectors'])}
                else:
                    # All possible current query detections; histories diverge across variants.
                    for b in boxes[scores>=.1]:s.observation(b)
        detections=_DetSlice(boxes,scores,classes)
        values={n:t.update_query_frame(detections,active) for n,t in trackers.items()}
        assert np.array_equal(values['native'],values['zero']);assert_states_equal(trackers['native'],trackers['zero'])
        if active:
            for name,s in servers.items():
                s.finish_frame()
                if not audit:
                    keys=list(s.current_ledger)
                    ledger[f'{frame}_{name}_boxes']=np.asarray([k[1:] for k in keys],np.float32).reshape(-1,4)
                    ledger[f'{frame}_{name}_vectors']=np.asarray([s.current_ledger[k] for k in keys],np.float32).reshape(-1,256)
        for name,value in values.items():
            if len(value):raw[name].append(np.column_stack([np.full(len(value),frame),value[:,:7]]).astype(np.float32))
            if active:
                anchor=anchors[frame]
                rows[name].append(scalabel_frame(video,dict(name=anchor['reference_name'],frameIndex=anchor['reference_index']),value))
    if capture is not None:capture.release()
    arrays={n:np.concatenate(x) if x else np.zeros((0,8),np.float32) for n,x in raw.items()}
    for name,t in trackers.items():
        if audit:
            assert np.array_equal(arrays[name],expected[name]),(name,'raw')
            assert rows[name]==service.read(gpu/(name+'.json')),(name,'JSON')
            assert t.events==service.read(gpu/(name+'_events.json')),(name,'cost')
            assert t.write_events==service.read(gpu/(name+'_writes.json')),(name,'memory')
        else:
            service.write(gpu/(name+'.json'),rows[name]);service.write(gpu/(name+'_events.json'),t.events)
            service.write(gpu/(name+'_writes.json'),t.write_events)
    if not audit:
        assert all(torch.equal(frozen[n][k],v) for n,m in models.items() for k,v in m.named_buffers())
        np.savez_compressed(gpu/'raw.npz',**arrays);np.savez_compressed(gpu/'vectors.npz',**ledger)
        files=[f for f in gpu.iterdir() if f.suffix in ('.json','.npz') and f.name not in ('protocol.json','execution_deadline.json','status.json','launch.json','controller.launch.json')]
        service.write(gpu/'tracking_completed.json',dict(complete=True,video=video,sha256={str(f.relative_to(ROOT)):service.sha(f) for f in files},updated=service.now()))
    metrics={}
    for name in trackers:
        status(stage='Development metrics only; detector has seen train',variant=name)
        dest=folder/'metric_inputs'/name;service.write(dest/(video+'.json'),rows[name])
        metrics[name]=evaluate_hota(dest,[video],split='train')
        if audit:assert metrics[name]==service.read(gpu/'report.json')['metrics'][name]
    if not audit:lock.__exit__(None,None,None)
    else:assert not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(complete=True,passed=True,technical_gate_only=True,
        source_video=video,variants=list(trackers),metrics=metrics,seconds=time.monotonic()-began,
        full_raw_json_cost_memory_metrics_replay=audit,scientific_verdict=None,
        limitation='One full fit20 development video; detector saw train; no detector generalization or scientific memory benefit claim. No tuning or variant selection from this result.',updated=service.now()))
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
