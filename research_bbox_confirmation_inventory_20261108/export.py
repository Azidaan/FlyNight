"""Export missing native old/new-fragment context without the legacy two-box/grid gate."""
import os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import deque,Counter
from research_bbox_confirmation_inventory_20261108.common import *


def export(video,entry,obs,p):
    import numpy as np
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import make_tracker_args
    from research_identity_assignment_run_20261108.replay import Detections
    from research_identity_assignment_run_20261108.full_planner import Planner
    from tracking.z13_region_assignment import regions
    from tracking.foveated_eyes import crop_box
    from ultralytics.trackers.utils import matching
    dest=RUN/'native'/video;marker=dest.with_suffix('.receipt.json')
    if marker.exists():
        r=read(marker)
        for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
        return r
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
    with np.load(ROOT/entry['raw'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    with np.load(ROOT/entry['baseline'],allow_pickle=False) as z:baseline=z['outputs'];frames=int(z['frame_count'])
    planner=Planner(video,'G');context={};seen=set();events=[];flat=[];counts=Counter();began=time.perf_counter()
    class Tracker(XYWHByteTrack):
        def _pre_first_associate(self,pool,unconfirmed,img,high):
            super()._pre_first_associate(pool,unconfirmed,img,high);planner.before(self,pool)
            context.update(high=high,unconfirmed={int(t.track_id):dict(frame=int(t.frame_id)-1,box=np.asarray(t.xyxy,float).tolist()) for t in unconfirmed})
        def _first_association(self,pool,detections,activated,refind):
            C=self.get_dists(pool,detections);matches,rr,cc=matching.linear_assignment(C,thresh=self.args.match_thresh);context.update(C=C,rr=list(map(int,rr)),cc=list(map(int,cc)),rows=planner.rows(pool) if context['unconfirmed'] else None)
            return super()._first_association(pool,detections,activated,refind)
    tracker=Tracker(make_tracker_args(**p['native_config']));order=np.argsort(raw['frame_idx'],kind='stable');sorted_frames=raw['frame_idx'][order]
    for frame in range(frames):
        lo,hi=np.searchsorted(sorted_frames,[frame,frame+1]);ids=order[lo:hi];out=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids));public={int(row[4]) for row in out};fresh=set(context['unconfirmed']) & public-seen;current={int(t.track_id):t for t in tracker.tracked_stracks};newcols={int(current[t].idx):t for t in fresh};high=context['high'];rows=[]
        if fresh:
            counts['native_first_public_confirmations']+=len(fresh)
            for j,t in newcols.items():assert context['unconfirmed'][t]['frame']==frame-1
            for i in context['rr']:
                row=context['rows'][i]
                if row['state']!=2 or not row['history'] or row['tid'] not in seen:continue
                age=frame-row['history'][-1]['frame']
                if age>p['max_age']:continue
                same=[j for j in context['cc'] if int(high.cls[j])==row['cls']];candidates=[]
                for j in same:
                    C=float(context['C'][i,j]);confidence=float(high.conf[j]);iou=float(np.clip((1-C)/confidence,0,1))
                    if C>=1 or iou<p['minimum_export_iou']:continue
                    tid=newcols.get(j);candidates.append(dict(column=j,raw_id=int(high.raw_ids[j]),box=high.xyxy[j].tolist(),confidence=confidence,cost=C,forward_iou=iou,new_tid=tid,birth=None if tid is None else context['unconfirmed'][tid]))
                if not any(c['new_tid'] is not None for c in candidates):continue
                choices,_,_=regions(planner.geom,row['current_crop'],np.asarray(row['mean'][:2]),high.xyxy,high.conf,high.cls,row['cls']);supported=len(choices)-1;rows.append(dict(row=i,old_tid=row['tid'],row_data=row,age=age,candidates=candidates,legacy_supported_box_count=supported));counts['old_rows_with_possible_first_public_link']+=1
                eligible=[c for c in candidates if c['forward_iou']>=p['inventory_iou']]
                if len(eligible)==1 and eligible[0]['new_tid'] is not None:counts['sole_bbox_at_inventory_iou']+=1;counts['sole_bbox_legacy_k'+str(supported)]+=1
            if rows:events.append(dict(frame=frame,newly_confirmed=sorted(fresh),rows=rows))
        if len(out):flat.append(np.column_stack([np.full(len(out),frame),out]).astype(np.float32))
        seen.update(public)
        for tid,t in current.items():
            if t.frame_id!=tracker.frame_id:continue
            planner.histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=frame,box=np.asarray(t.xyxy,float).tolist(),crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))));planner.searches[tid]=[]
        if frame%300==0:status('bbox_native_missing_context',video=video,split=entry['split'],frame=frame,total=frames,events=len(events),guard=obs.read())
    values=np.concatenate(flat) if flat else np.empty((0,9),np.float32);assert np.array_equal(values[:,:8],baseline),(video,'Native complete output changed')
    dest.parent.mkdir(parents=True,exist_ok=True);npz=dest.with_suffix('.npz');meta=dest.with_suffix('.json')
    with npz.open('xb') as f:np.savez_compressed(f,outputs=values,frame_count=frames)
    write(meta,dict(passed=True,video=video,split=entry['split'],frames=frames,events=events,counts=dict(counts),complete_cached_native_output_exact=True),new=True)
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h,n
    r=dict(passed=True,video=video,split=entry['split'],frames=frames,counts=dict(counts),seconds=time.perf_counter()-began,artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [npz,meta]});write(marker,r,new=True);return r


def main(obs):
    p=phase();results={};totals=Counter()
    for video,entry in p['videos'].items():r=export(video,entry,obs,p);results[video]=r;totals.update(r['counts'])
    phase();write(RUN/'inventory.json',dict(passed=True,videos=results,totals=dict(totals),new_core_calls=0,new_detector_calls=0,new_image_reads=0,scope='Missing native context and exposure only; no labels or association-quality claim. Complete native output equals each ready baseline.'),new=True);status('bbox_context_inventory_complete',videos=len(results),totals=dict(totals))


if __name__=='__main__':main(guard())
