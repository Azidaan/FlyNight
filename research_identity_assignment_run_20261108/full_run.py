"""One full closed-loop rollout; only observable detections, clocks and past images."""
import gzip,time
import numpy as np
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.utils import matching,stracks as pool_helpers
from tracking.spatial_identity_assignment import SpatialIdentityByteTrack,corrected_cost
from tracking.offline_association import make_tracker_args
from datasets.bdd100k_loader import CLASS_NAMES
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.replay import Detections
from research_identity_assignment_run_20261108.full_planner import Planner,Representations
from research_identity_assignment_run_20261108.full_retina import FULL


class LiveTracker(SpatialIdentityByteTrack):
    def __init__(self,args,planner,**kw):
        super().__init__(args,provider=planner.provide,**kw);self.planner=planner;self.assigned={};self.dupes=[]
    def init_track(self,results,img=None):
        tracks=super().init_track(results,img)
        for t,raw in zip(tracks,results.raw_ids):t.export_raw_id=int(raw)
        return tracks
    def _apply_match(self,t,d,a,r):
        super()._apply_match(t,d,a,r);t.export_raw_id=d.export_raw_id;self.assigned[int(t.track_id)]=int(d.export_raw_id)
    def _pre_first_associate(self,pool,unconfirmed,img,high):
        super()._pre_first_associate(pool,unconfirmed,img,high);self.planner.high=high;self.planner.before(self,pool)
    def update(self,*a,**kw):
        self.assigned={};self.dupes=[];original=pool_helpers.remove_duplicate_stracks
        def dedup(first,second,dup_thresh=.15):
            distance=matching.iou_distance(first,second);aa,bb=original(first,second,dup_thresh)
            ka={x.track_id for x in aa};kb={x.track_id for x in bb}
            for i,j in zip(*np.where(distance<dup_thresh)):
                self.dupes.append(dict(tracked_id=int(first[i].track_id),lost_id=int(second[j].track_id),
                    iou=float(1-distance[i,j]),tracked_dropped=first[i].track_id not in ka,lost_dropped=second[j].track_id not in kb))
            return aa,bb
        pool_helpers.remove_duplicate_stracks=dedup
        try:value=super().update(*a,**kw)
        finally:pool_helpers.remove_duplicate_stracks=original
        self.planner.after(self);return value


def scalabel(video,clock,output):
    labels=[]
    for row in output:
        labels.append(dict(id=str(int(row[4])),category=CLASS_NAMES[int(row[6])],score=float(row[5]),
            box2d=dict(zip(['x1','y1','x2','y2'],map(float,row[:4])))))
    return dict(videoName=video,name=clock['name'],frameIndex=clock['frameIndex'],labels=labels)


def rollout(video,arm,obs,*,backbone=None,zero=False):
    p=read(FULL/'protocol.json');entry=p['videos'][video];dest=FULL/'rollouts'/arm;dest.mkdir(parents=True,exist_ok=True)
    marker=dest/(video+'.receipt.json');rawpath=dest/(video+'.npz');jsonpath=FULL/'outputs'/arm/(video+'.json')
    if marker.exists():
        receipt=read(marker)
        for n,h in receipt['artifacts'].items():assert sha(ROOT/n)==h
        return receipt
    assert not rawpath.exists() and not jsonpath.exists()
    clock=read(FULL/'runtime_inputs'/(video+'.json'));anchors=clock['anchors']
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    assert sha(ROOT/entry['detections'])==entry['sources'][entry['detections']]
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order]
    settings=dict(alpha=0,beta=0) if zero else p['parameters'][arm]
    rep=None
    if settings['beta']!=0:rep=Representations(video,'R1' if arm=='G_R1' else 'C',backbone)
    planner=Planner(video,arm,rep);tracker=LiveTracker(make_tracker_args(**p['native_config']),planner,arm=arm,**settings)
    outputs=[];rows=[];events=[];began=time.perf_counter()
    for f in range(entry['frames']):
        lo,hi=np.searchsorted(frames,[f,f+1]);ids=order[lo:hi]
        detections=Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids)
        output=tracker.update(detections)
        if len(output):outputs.append(np.column_stack([np.full(len(output),f),np.asarray(output)[:,:7]]).astype(np.float32))
        if str(f) in anchors:rows.append(scalabel(video,anchors[str(f)],output))
        if planner.last is not None and planner.last['frame']==f and planner.last['plan']['votes']:
            snap=planner.last;C=snap['cost'];plan=snap['plan'];native=np.asarray(matching.linear_assignment(C,thresh=.8)[0]).reshape(-1,2).tolist()
            emitted={int(r[4]) for r in output}
            events.append(dict(frame=f,rows=snap['rows'],plan=plan,C=C.tolist(),
                corrected=corrected_cost(C,plan['votes'],settings['alpha'],settings['beta'],arm,plan['columns']).tolist(),
                native_first=native,actual_first=tracker.identity_events[-1]['matches'],
                emitted_ids=sorted(emitted),assigned={str(k):v for k,v in tracker.assigned.items() if k in emitted},duplicates=tracker.dupes))
        if f%50==0:status('full_closed_loop',video=video,arm=arm,frame=f,total=entry['frames'],events=len(events),guard=obs.read())
    flat=np.concatenate(outputs) if outputs else np.zeros((0,8),np.float32)
    if zero:
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:
            assert int(z['frame_count'])==entry['frames'] and np.array_equal(flat,z['outputs']),'Full val zero mismatch'
        assert rows==read(ROOT/entry['baseline']),'Baseline converter mismatch'
    with rawpath.open('xb') as f:np.savez_compressed(f,outputs=flat,frame_count=entry['frames'])
    write(jsonpath,rows,new=True);eventpath=dest/(video+'.events.json.gz')
    with gzip.open(eventpath,'xt',encoding='utf-8') as f:json.dump(events,f,ensure_ascii=False,allow_nan=False)
    receipt=dict(passed=True,video=video,arm=arm,zero=zero,frames=entry['frames'],seconds=time.perf_counter()-began,
        artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [rawpath,jsonpath,eventpath]},
        feature_files={} if rep is None else rep.files,
        core_imported=any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules))
    if arm in ['G','G_R1','B0']:assert not receipt['core_imported']
    write(marker,receipt,new=True);return receipt
