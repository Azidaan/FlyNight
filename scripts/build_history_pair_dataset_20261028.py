"""Реальные первые назначения и причинные истории для матрицы1/3/4, CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from collections import defaultdict,deque
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    import numpy as np
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from tracking.history_training_data import label_detection_identities,identity_target,pair_geometry
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,CLASS_NAMES
    folder=RUN/'history_pair_dataset';p=read(folder/'protocol.json')
    pre=RUN/'history_dataset_checks';assert read(pre/'report.json')['passed']
    pc=read(pre/'controller_result.json');assert pc['exit_code']==0 and pc['all_owned_ended'] and not any(service.matching(k,v) for k,v in pc['owned'].items())
    previous=RUN/'history_observed_export';c=read(previous/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    exported=read(previous/'report.json');assert exported['complete']
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp();completed=[]
    for vi,src in enumerate(exported['completed']):
        started=time.monotonic();v=src['video'];outfolder=folder/'videos'/v
        marker=outfolder/'report.json'
        if marker.exists():
            old=read(marker)
            assert sha(outfolder/'inputs.npz')==old['inputs_sha256'] and sha(outfolder/'targets.npz')==old['targets_sha256'];completed.append(old);continue
        outfolder.mkdir(parents=True,exist_ok=True)
        assert sha(ROOT/src['file'])==src['sha256']
        with np.load(ROOT/src['file']) as z:observed={k:z[k] for k in z.files}
        cache_path=ROOT/f'runs/train_cache/{v}/detections/raw_video.npz'
        assert sha(cache_path)==src['source_sha256']
        cache=load_detection_cache(cache_path)
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json');alignment=read(ROOT/f'runs/train_cache/{v}/alignment.json')
        schedule={int(x['video_frame_index']):int(x['reference_index']) for x in alignment['matches']}
        assert len(schedule)==len(alignment['matches'])==len(truth)
        assert all(int(b['video_frame_index'])>int(a['video_frame_index']) for a,b in zip(alignment['matches'][:-1],alignment['matches'][1:]))
        assert all(truth[int(x['reference_index'])]['name']==x['reference_name'] for x in alignment['matches'])
        history=defaultdict(lambda:deque(maxlen=p['history_items']));rows=[];ys=[];knowns=[]
        class Capture(XYWHByteTrack):
            def _first_association(self,*args):
                self.first=True
                try:return super()._first_association(*args)
                finally:self.first=False
            def get_dists(self,tracks,detections):
                matrix=super().get_dists(tracks,detections)
                if not self.first or self.frame not in schedule:return matrix
                for i,track in enumerate(tracks):
                    tid=int(track.track_id);past=[x for x in history[tid] if 0<self.frame-x['frame']<=p['max_history_frames']]
                    if not past:continue
                    for j,det in enumerate(detections):
                        if matrix[i,j]>self.args.match_thresh or int(det.cls)!=int(track.cls):continue
                        idx=int(det.idx);box=self.boxes[idx];target,known=identity_target(past[-1]['identity'],self.identities[idx])
                        assert all(x['frame']<self.frame for x in past)
                        rows.append(dict(frame=self.frame,track=tid,det=idx,box=box.copy(),geometry=pair_geometry(track,box,float(self.scores[idx]),float(matrix[i,j]),self.frame,len(past)),history=past.copy()))
                        ys.append(target);knowns.append(known)
                return matrix
        tracker=Capture(SimpleNamespace(**p['tracker_config']));tracker.reset_id();tracker.first=False
        for frame in range(int(cache['frame_count'])):
            if time.time()>end-10:raise TimeoutError('History pair dataset budget')
            boxes,scores,classes=detections_for_frame(cache,frame)
            ids=label_detection_identities(boxes,classes,truth[schedule[frame]]['labels'],CLASS_NAMES) if frame in schedule else [None]*len(boxes)
            tracker.frame=frame;tracker.boxes=boxes;tracker.scores=scores;tracker.identities=ids
            out=np.asarray(tracker.update(_DetSlice(boxes,scores,classes)),np.float32).reshape(-1,8)
            selected=np.flatnonzero(observed['frame']==frame)
            assert np.array_equal(out[:,4].astype(np.int32),observed['track_id'][selected])
            assert np.array_equal(out[:,7].astype(np.int32),observed['detection_index'][selected])
            assert np.array_equal(out[:,:4],observed['filtered_xyxy'][selected])
            if frame in schedule:
                for row in out:
                    idx=int(row[7]);history[int(row[4])].append(dict(frame=frame,det=idx,box=boxes[idx].copy(),quality=float(scores[idx]),identity=ids[idx]))
            if frame%100==0:write(folder/'status.json',dict(state='running',video=v,index=vi,total=len(exported['completed']),frame=frame,total_frames=int(cache['frame_count']),pairs=len(rows),updated=now()),required=False)
        n=len(rows);K=p['history_items'];hf=np.full((n,K),-1,np.int32);hd=hf.copy();hb=np.zeros((n,K,4),np.float32);hq=np.zeros((n,K),np.float32);valid=np.zeros((n,K),bool)
        for i,r in enumerate(rows):
            for j,h in enumerate(r['history']):hf[i,j]=h['frame'];hd[i,j]=h['det'];hb[i,j]=h['box'];hq[i,j]=h['quality'];valid[i,j]=True
        np.savez_compressed(outfolder/'inputs.npz',frame=np.array([r['frame'] for r in rows],np.int32),track=np.array([r['track'] for r in rows],np.int32),det=np.array([r['det'] for r in rows],np.int32),
            box=np.asarray([r['box'] for r in rows],np.float32).reshape(n,4),geometry=np.asarray([r['geometry'] for r in rows],np.float32).reshape(n,14),history_frame=hf,history_det=hd,history_box=hb,history_quality=hq,history_valid=valid)
        np.savez_compressed(outfolder/'targets.npz',identity=np.array(ys,np.int8),known=np.array(knowns,bool))
        assert all(y==-1 for y,k in zip(ys,knowns) if not k)
        entry=dict(video=v,split=src['split'],pairs=n,known=int(sum(knowns)),positive=sum(y==1 for y in ys),unknown=n-int(sum(knowns)),
            seconds=time.monotonic()-started,inputs_sha256=sha(outfolder/'inputs.npz'),targets_sha256=sha(outfolder/'targets.npz'),
            observation_source_sha256=src['sha256'],alignment_sha256=sha(ROOT/f'runs/train_cache/{v}/alignment.json'),gt_source_sha256=sha(ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json'),gt_only_targets=True,baseline_replay_exact=True)
        write(marker,entry);completed.append(entry);write(folder/'completed_videos.json',completed)
        if vi==0:
            need=2*entry['seconds']*(len(exported['completed'])-1)+120;available=end-time.time();price=dict(first_full_video_seconds=entry['seconds'],conservative_remaining_seconds=need,available_seconds=available,proceed=need<available);write(folder/'price.json',price)
            if not price['proceed']:
                write(folder/'report.json',dict(complete=False,stopped_on_cost=True,completed=completed,remaining=exported['completed'][1:],hota_measured=False,training_ready=False,updated=now()));return
        # Записи GT не входят в сохранённые входы модели; только массивы targets.
    write(folder/'report.json',dict(complete=True,completed=completed,pairs=sum(x['pairs'] for x in completed),known=sum(x['known'] for x in completed),
        gt_only_targets=True,baseline_replay_exact=True,hota_measured=False,training_ready=False,updated=now()))

if __name__=='__main__':main()
