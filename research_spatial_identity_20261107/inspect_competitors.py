"""Recover missing competitor histories, without changing native association."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write,verified_protocol,guard,status


def main(obs):
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice,make_tracker_args
    from ultralytics.trackers.basetrack import BaseTrack
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.output_filter_oracle import _iou
    verified_protocol();phase=read(RUN/'competitor_phase_protocol.json')
    for name,h in phase['source_sha256'].items():assert sha(ROOT/name)==h,name
    video='0001542f-ec815219';ids={158,196,189,276,274};torch.set_num_threads(2)
    source=read(RUN/'tracker'/(video+'.json'))
    raw=ROOT/'runs/train_cache'/video/'detections/raw_video.npz'
    with np.load(raw,allow_pickle=False) as z:boxes=z['xyxy'];scores=z['conf'];classes=z['cls'];frames=z['frame_idx']
    alignment=read(ROOT/'runs/train_cache'/video/'alignment.json')['matches']
    annotation=read(ROOT/'data/bdd100k/labels/box_track_20/train'/(video+'.json'))
    truth={int(a['video_frame_index']):b['labels'] for a,b in zip(alignment,annotation)}
    context={'frame':0};assignments={};traces={str(i):[] for i in ids}
    class Detections(_DetSlice):
        def __init__(self,b,s,c,raw_ids):super().__init__(b,s,c);self.raw_ids=np.array(raw_ids)
        def __getitem__(self,mask):return Detections(self._xyxy[mask],self.conf[mask],self.cls[mask],self.raw_ids[mask])
    class TraceTracker(XYWHByteTrack):
        def init_track(self,results,img=None):
            tracks=super().init_track(results,img)
            for t,i in zip(tracks,results.raw_ids):t.export_raw_id=int(i)
            return tracks
        def _apply_matches(self,matches,tracks,detections,activated,refind):
            for i,j in matches:assignments[(context['frame'],int(tracks[i].track_id))]=int(detections[j].export_raw_id)
            return super()._apply_matches(matches,tracks,detections,activated,refind)
    BaseTrack.reset_id();tracker=TraceTracker(make_tracker_args(track_high_thresh=.25,track_low_thresh=.1,track_buffer=30,match_thresh=.8))
    order=np.argsort(frames,kind='stable');ordered=frames[order]
    last_annotation={};query_frames={332,440}
    for f in range(441):
        context['frame']=f;lo,hi=np.searchsorted(ordered,[f,f+1]);ix=order[lo:hi]
        tracker.update(Detections(boxes[ix],scores[ix],classes[ix],ix))
        active=[t for t in tracker.tracked_stracks if t.frame_id==tracker.frame_id and t.is_activated]
        linked={}
        if f in truth:
            for name in set(CLASS_NAMES.values()):
                ts=[t for t in active if CLASS_NAMES.get(int(t.cls))==name]
                gs=[g for g in truth[f] if g['category']==name and not g.get('attributes',{}).get('crowd',False)]
                if not ts or not gs:continue
                bb=[[g['box2d'][k] for k in ['x1','y1','x2','y2']] for g in gs]
                matrix=_iou([t.xyxy for t in ts],bb);ii,jj=linear_sum_assignment(1-matrix)
                for i,j in zip(ii,jj):
                    if matrix[i,j]>=.5:linked[int(ts[i].track_id)]={'gt_id':str(gs[j]['id']),'iou':float(matrix[i,j]),'frame':f}
            for t in active:
                tid=int(t.track_id)
                if tid in ids:last_annotation[tid]=linked.get(tid,{'gt_id':None,'frame':f})
        for t in active:
            tid=int(t.track_id)
            if tid not in ids:continue
            traces[str(tid)].append(dict(frame=f,box=t.xyxy.tolist(),raw_id=assignments.get((f,tid),int(getattr(t,'export_raw_id',-1))),
                annotation=linked.get(tid),last_annotated_match=last_annotation.get(tid)))
        if f%100==0:status('competitor_history_export',frame=f,total=441,guard=obs.read())
    checks=[]
    for r in source['rows']:
        actual=assignments.get((r['frame'],r['tid']))
        assert actual==r['native_assignment_raw_id']
        checks.append(dict(uid=r['uid'],native_assignment_equal=True,frame=r['frame']))
    with np.load(ROOT/'runs/z13_data'/(video+'.npz'),allow_pickle=False) as z:
        targets={r['uid']:str(z['ep_identity'][r['episode']]) for r in source['rows']}
    summary=[]
    for r in source['rows']:
        relevant=[r['tid']]+[i for i in r['candidate_assigned_track_ids'] if i is not None]
        items={}
        for tid in relevant:
            trace=[x for x in traces[str(tid)] if x['frame']<=r['frame']]
            marked=[x['annotation'] for x in trace if x['annotation']]
            items[str(tid)]={'last_observed_frame':trace[-1]['frame'] if trace else None,
                'last_annotated_match':trace[-1]['last_annotated_match'] if trace else None,
                'known_gt_ids':dict(Counter(x['gt_id'] for x in marked)),
                'last_five_annotations':marked[-5:]}
        summary.append(dict(uid=r['uid'],frame=r['frame'],original_target_gt_id=targets[r['uid']],tracks=items))
    assert not torch.cuda.is_initialized()
    write(RUN/'competitor_histories.json',dict(passed=True,video=video,checks=checks,summary=summary,traces=traces,
        limitation='Sparse annotated-frame Hungarian IoU>=0.5 identities are descriptive. They cannot certify identity between annotations or justify stealing a detection from a live track.'),new=True)
    print(summary)


if __name__=='__main__':main(guard())
