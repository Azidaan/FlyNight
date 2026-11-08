"""Restore a native pre-update state and apply one causal full lifecycle update."""
import pickle
import numpy as np
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.bot_sort import BOTrack
from tracking.offline_association import _DetSlice,make_tracker_args
from tracking.spatial_identity_assignment import SpatialIdentityByteTrack
from ultralytics.trackers.utils import stracks as pool_helpers,matching


class Detections(_DetSlice):
    def __init__(self,b,s,c,ids):super().__init__(b,s,c);self.raw_ids=np.asarray(ids)
    def __getitem__(self,mask):return Detections(self._xyxy[mask],self.conf[mask],self.cls[mask],self.raw_ids[mask])


class ReplayTracker(SpatialIdentityByteTrack):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.assigned={}
    def init_track(self,results,img=None):
        tracks=super().init_track(results,img)
        for t,i in zip(tracks,results.raw_ids):t.export_raw_id=int(i)
        return tracks
    def _apply_match(self,t,det,activated,refind):
        value=super()._apply_match(t,det,activated,refind)
        t.export_raw_id=det.export_raw_id
        self.assigned[int(t.track_id)]=int(det.export_raw_id);return value


def apply_state(blob,cfg,frame_inputs,snapshot,plan,alpha,beta,arm):
    data=pickle.loads(blob)
    def provider(tracker,pool,detections,cost):
        assert [int(t.track_id) for t in pool]==[r['tid'] for r in snapshot['rows']]
        assert [d.export_raw_id for d in detections]==snapshot['raw_ids']
        if '_native_cost' in plan:assert np.array_equal(cost,plan['_native_cost'])
        return plan
    tracker=ReplayTracker(make_tracker_args(**cfg),alpha=alpha,beta=beta,arm=arm,provider=provider)
    # The constructor resets the global counter. Restore it afterwards.
    BaseTrack._count=data['count'];tracker.frame_id=data['frame_id'];objects={}
    for name,rows in data['pools'].items():
        pool=[]
        for attributes in rows:
            key=id(attributes)
            if key not in objects:
                t=BOTrack.__new__(BOTrack);t.__dict__=attributes;objects[key]=t
            pool.append(objects[key])
        setattr(tracker,name,pool)
    b,s,c,ids=frame_inputs
    duplicate_events=[];original_dedup=pool_helpers.remove_duplicate_stracks
    def audited_dedup(a,b,dup_thresh=.15):
        distances=matching.iou_distance(a,b)
        aa,bb=original_dedup(a,b,dup_thresh)
        kept_a={t.track_id for t in aa};kept_b={t.track_id for t in bb}
        for i,j in zip(*np.where(distances<dup_thresh)):
            duplicate_events.append(dict(tracked_id=int(a[i].track_id),lost_id=int(b[j].track_id),
                tracked_duration=int(a[i].frame_id-a[i].start_frame),lost_duration=int(b[j].frame_id-b[j].start_frame),
                iou=float(1-distances[i,j]),tracked_dropped=a[i].track_id not in kept_a,lost_dropped=b[j].track_id not in kept_b))
        return aa,bb
    pool_helpers.remove_duplicate_stracks=audited_dedup
    try:output=tracker.update(Detections(b,s,c,ids))
    finally:pool_helpers.remove_duplicate_stracks=original_dedup
    high=ids[s>=cfg['track_high_thresh']]
    final={};all_final={}
    emitted={int(row[4]) for row in output}
    for t in tracker.tracked_stracks:
        tid=int(t.track_id)
        if t.frame_id!=tracker.frame_id:continue
        raw=tracker.assigned[tid] if tid in tracker.assigned else int(high[int(t.idx)])
        all_final[str(tid)]=raw
        if tid in emitted:final[str(tid)]=raw
    return dict(output=output.tolist(),final=final,all_final=all_final,duplicate_events=duplicate_events,
                first=tracker.identity_events[-1]['matches'] if tracker.identity_events else snapshot['native_matches'])
