"""Реальные назначения исходного трекера из имеющихся детекций; только CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now

def main():
    import numpy as np
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    folder=RUN/'history_observed_export';p=read(folder/'protocol.json')
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    previous=RUN/'history_sources_after_partial';c=read(previous/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    assert read(previous/'report.json')['preparation_completed']
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    (folder/'videos').mkdir(exist_ok=True);(folder/'completed').mkdir(exist_ok=True)
    done=[]
    for index,source in enumerate(p['sources']):
        v=source['video'];marker=folder/'completed'/(v+'.json')
        if marker.exists():
            m=read(marker);assert sha(ROOT/m['file'])==m['sha256'];done.append(m);continue
        started=time.monotonic();cache_path=ROOT/source['detections'];cache_hash=sha(cache_path)
        cache=load_detection_cache(cache_path)
        tracker=XYWHByteTrack(SimpleNamespace(**p['tracker_config']),frame_rate=30);tracker.reset_id()
        frames=[];tracks=[];indices=[];boxes=[];filtered=[];scores=[];classes=[]
        for frame in range(int(cache['frame_count'])):
            if time.time()>end-10:raise TimeoutError('Сохранить экспорт в прежнем бюджете; готовые видео не повторять')
            b,confidence,labels=detections_for_frame(cache,frame)
            out=np.asarray(tracker.update(_DetSlice(b,confidence,labels)),np.float32).reshape(-1,8)
            active={int(t.track_id):t for t in tracker.tracked_stracks if t.is_activated}
            seen=set()
            for row in out:
                tid=int(row[4]);det=int(row[7]);assert tid not in seen;seen.add(tid)
                assert int(active[tid].frame_id)==frame+1
                assert 0<=det<len(b) and float(row[5])==float(confidence[det]) and int(row[6])==int(labels[det])
                assert np.all(np.asarray(b[det])[2:]>np.asarray(b[det])[:2])
                frames.append(frame);tracks.append(tid);indices.append(det);boxes.append(b[det]);filtered.append(row[:4]);scores.append(confidence[det]);classes.append(labels[det])
            if frame%100==0:write(folder/'status.json',dict(state='running',video=v,index=index,total=len(p['sources']),frame=frame,total_frames=int(cache['frame_count']),updated=now()),required=False)
        target=folder/'videos'/(v+'.npz')
        np.savez_compressed(target,frame=np.asarray(frames,np.int32),track_id=np.asarray(tracks,np.int32),detection_index=np.asarray(indices,np.int32),
            observed_xyxy=np.asarray(boxes,np.float32).reshape(-1,4),filtered_xyxy=np.asarray(filtered,np.float32).reshape(-1,4),
            score=np.asarray(scores,np.float32),cls=np.asarray(classes,np.int16))
        with np.load(target) as z:
            assert len(z['frame'])==len(frames) and np.array_equal(z['observed_xyxy'],np.asarray(boxes,np.float32).reshape(-1,4))
        m=dict(video=v,split=source['split'],file=target.relative_to(ROOT).as_posix(),sha256=sha(target),
            source_sha256=cache_hash,observations=len(frames),seconds=time.monotonic()-started,ground_truth_used=False,updated=now())
        write(marker,m);done.append(m)
        if index==0:
            need=2*m['seconds']*(len(p['sources'])-1)+60;available=end-time.time()
            price=dict(first_full_video_seconds=m['seconds'],conservative_remaining_seconds=need,available_seconds=available,proceed=need<available)
            write(folder/'price.json',price)
            if not price['proceed']:
                write(folder/'report.json',dict(complete=False,stopped_on_cost=True,completed=done,remaining=p['sources'][1:],ready_for_training=False,hota_measured=False,scientific_verdict=None,updated=now()));return
    write(folder/'report.json',dict(complete=True,completed=done,ready_for_training=False,hota_measured=False,scientific_verdict=None,
        limitation='Только реально выданные активированные треки и их текущие детекции; рабочий ID не истинная личность. Изображения/признаки/совмещение/обучение не экспортируются этим шагом.',updated=now()))

if __name__=='__main__':main()
