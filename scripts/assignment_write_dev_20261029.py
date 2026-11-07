"""One new write-admission action on exact existing development vectors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from types import SimpleNamespace
import sys,time,faulthandler
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import numpy as np,torch
    from tracking.query_assignment_write import AssignmentWriteTracker
    from tracking.query_appearance_comparison import QueryAppearanceTracker,assert_states_equal
    from tracking.frame_masa_appearance import FrameMasaServer
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    audit='--audit' in sys.argv
    out=RUN/'assignment_write_dev';folder=RUN/('assignment_write_dev_audit' if audit else 'assignment_write_dev')
    p=service.read(out/'protocol.json');source=p['source'];old=RUN/'memory_write_dev';video=source['video']
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    if audit:
        c=service.read(out/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended']
        assert not any(service.matching(k,v) for k,v in c['owned'].items())
        assert all(service.sha(ROOT/f)==h for f,h in service.read(out/'tracking_completed.json')['sha256'].items())
    else:assert not (out/'tracking_completed.json').exists(),'Recover metrics only; do not replay completed new output'
    torch.set_default_device('cpu');torch.set_num_threads(2)
    deadline=datetime.fromisoformat(service.read(folder/'execution_deadline.json')['deadline']).timestamp()
    class Replay(FrameMasaServer):
        def features(self,tokens,null_seed=None):
            assert null_seed is None
            for t in tokens:
                assert t.frame<=self.frame
                if t.vector is None:
                    assert t.frame==self.frame;t.vector=self.values[tuple(t.crop.tolist())].copy()
            return np.stack([t.vector for t in tokens])
    servers={n:Replay(None) for n in ['r50','detic']};args=SimpleNamespace(**p['config'])
    trackers={'native_shadow':QueryAppearanceTracker(args),'zero':AssignmentWriteTracker(args,servers['detic'],p['appearance_weight'],True)}
    trackers.update({n+'_guarded':AssignmentWriteTracker(args,s,p['appearance_weight']) for n,s in servers.items()})
    vectors=np.load(old/'vectors.npz');expected_native=np.load(old/'raw.npz')['native']
    cache=load_detection_cache(ROOT/source['detections']);align=service.read(ROOT/source['alignment'])['matches']
    anchors={int(x['video_frame_index']):x for x in align};raw={n:[] for n in trackers};rows={n:[] for n in trackers};began=time.monotonic()
    for frame in range(int(cache['frame_count'])):
        if frame%50==0:
            assert time.time()<deadline-30
            faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(600,exit=True)
            service.write(folder/'status.json',dict(stage='Independent CPU replay' if audit else 'New assignment-based write admission on exact dev vectors',frame=frame,total_frames=int(cache['frame_count']),video=video,updated=service.now()))
        active=frame in anchors
        if active:
            for n,s in servers.items():
                s.start_frame(frame,None);s.values={tuple(b.tolist()):v for b,v in zip(vectors[f'{frame}_{n}_boxes'],vectors[f'{frame}_{n}_vectors'])}
        boxes,scores,classes=detections_for_frame(cache,frame);d=_DetSlice(boxes,scores,classes)
        values={n:t.update_query_frame(d,active) for n,t in trackers.items()}
        assert np.array_equal(values['native_shadow'],values['zero']);assert_states_equal(trackers['native_shadow'],trackers['zero'])
        if active:
            for s in servers.values():s.finish_frame()
        for n,value in values.items():
            if len(value):raw[n].append(np.column_stack([np.full(len(value),frame),value[:,:7]]).astype(np.float32))
            if active:
                anchor=anchors[frame];rows[n].append(scalabel_frame(video,dict(name=anchor['reference_name'],frameIndex=anchor['reference_index']),value))
    arrays={n:np.concatenate(x) if x else np.zeros((0,8),np.float32) for n,x in raw.items()};assert np.array_equal(arrays['native_shadow'],expected_native)
    for n,t in trackers.items():
        if audit:
            assert np.array_equal(arrays[n],np.load(out/'raw.npz')[n]);assert rows[n]==service.read(out/(n+'.json'))
            assert t.events==service.read(out/(n+'_events.json'))
            assert getattr(t,'write_events',[])==service.read(out/(n+'_writes.json'))
        else:
            service.write(out/(n+'.json'),rows[n]);service.write(out/(n+'_events.json'),t.events);service.write(out/(n+'_writes.json'),getattr(t,'write_events',[]))
    if not audit:
        np.savez_compressed(out/'raw.npz',**arrays)
        files=[f for f in out.iterdir() if f.suffix in ('.json','.npz') and f.name not in ('protocol.json','execution_deadline.json','launch.json','controller.launch.json','status.json')]
        service.write(out/'tracking_completed.json',dict(sha256={str(f.relative_to(ROOT)):service.sha(f) for f in files},updated=service.now()))
    metrics={}
    for n in ['r50_guarded','detic_guarded']:
        dest=folder/'metric_inputs'/n;service.write(dest/(video+'.json'),rows[n]);metrics[n]=evaluate_hota(dest,[video],split='train')
        if audit:assert metrics[n]==service.read(out/'report.json')['metrics'][n]
    assert not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(complete=True,passed=True,technical_gate_only=True,metrics=metrics,video=video,seconds=time.monotonic()-began,
        zero_raw_state_exact=True,full_raw_json_events_memory_metrics_replay=audit,
        counts={n:dict(accepted=sum(e['accepted'] for e in t.write_events),rejected=sum(not e['accepted'] for e in t.write_events)) for n,t in trackers.items() if hasattr(t,'write_events')},
        limitation='One dev TRAIN video already seen by detector, no quality-based tuning/selection, not scientific memory benefit, not full KeepTrack. May retain stale or initially wrong memory.',updated=service.now()))
    faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':main()
