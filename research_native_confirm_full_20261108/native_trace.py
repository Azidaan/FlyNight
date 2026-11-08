"""Recover missing causal native-confirmation metadata, once, without an intervention."""
import os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import deque,Counter
from research_identity_assignment_run_20261108.common import read,write,sha,datetime,MSK
from research_singleton_verify_20261108.common import guard
from research_singleton_verify_full_recovery1_20261108.status_io import for_run
from research_native_confirm_full_20261108.common import FULL
RUN=FULL;status=for_run(RUN)


def trace(video,obs,protocol):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from ultralytics.trackers.basetrack import BaseTrack
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import make_tracker_args
    from research_identity_assignment_run_20261108.replay import Detections
    from research_identity_assignment_run_20261108.full_planner import Planner,dis_domain
    from tracking.spatial_identity_assignment import geometric_match,frame_points
    from tracking.own_template_verification import correspondence
    from tracking.z13_region_assignment import regions
    from tracking.foveated_eyes import crop_box
    dest=RUN/'native_trace'/video;receipt=dest.with_suffix('.receipt.json')
    if receipt.exists():
        d=read(receipt)
        for n,h in d['artifacts'].items():assert sha(ROOT/n)==h
        return d
    entry=protocol['videos'][video];p=ROOT/entry['raw'];old=read(ROOT/entry['export']) if entry.get('export') else None;oldmat=np.load((ROOT/entry['export']).with_suffix('.npz'),allow_pickle=False) if old else None
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    with np.load(p,allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order];total=entry.get('frames',int(frames.max())+1);planner=Planner(video,'G');context={};seen=set();events=[];flat=[];offsets=[0];checks=Counter();began=time.perf_counter()

    class Tracker(XYWHByteTrack):
        def _pre_first_associate(self,pool,unconfirmed,img,high):
            super()._pre_first_associate(pool,unconfirmed,img,high);planner.before(self,pool);context.update(pool=pool,unconfirmed={int(t.track_id) for t in unconfirmed},high=high)

        def _first_association(self,pool,detections,activated,refind):
            C=self.get_dists(pool,detections);native,rr,cc=matching.linear_assignment(C,thresh=self.args.match_thresh)
            context.update(C=C,native=np.asarray(native).reshape(-1,2).tolist(),rr=set(map(int,rr)),cc=set(map(int,cc)),rows=planner.rows(pool) if context['unconfirmed'] or old is not None and str(planner.frame) in old['snapshots'] else None)
            if old is not None and str(planner.frame) in old['snapshots']:
                saved=old['snapshots'][str(planner.frame)];assert np.array_equal(C,oldmat['cost_'+str(planner.frame)]);assert context['native']==saved['native_matches'];assert len(context['rows'])==len(saved['rows']);assert context['rows']==saved['rows'] or all(all(r[k]==s[k] for k in ['tid','state','cls','start_frame','mean','covariance','history','current_crop','search']) for r,s in zip(context['rows'],saved['rows']))
                checks['archived_native_state_and_cost_frames']+=1
            return super()._first_association(pool,detections,activated,refind)

    BaseTrack.reset_id();tracker=Tracker(make_tracker_args(**protocol['native_config']))
    for frame in range(total):
        lo,hi=np.searchsorted(frames,[frame,frame+1]);ix=order[lo:hi];out=tracker.update(Detections(raw['xyxy'][ix],raw['conf'][ix],raw['cls'][ix],ix));ids={int(v[4]) for v in out};fresh=context['unconfirmed'] & ids-seen;high=context['high'];highids=high.raw_ids.tolist();current={int(t.track_id):t for t in tracker.tracked_stracks};newcols={int(current[t].idx):t for t in fresh};edges=[]
        if old is not None and str(frame) in old['snapshots']:
            assert np.array_equal(out,np.asarray(old['snapshots'][str(frame)]['native_output'],dtype=np.float32).reshape(-1,8)),(video,frame,'native output mismatch');checks['archived_output_frames']+=1
        if fresh:
            checks['native_first_public_confirmations']+=len(fresh)
            for i in sorted(context['rr']):
                row=context['rows'][i]
                if row['state']!=2 or not row['history'] or row['tid'] not in seen:continue
                possible=[j for j in newcols if j in context['cc'] and context['C'][i,j]<1 and row['cls']==int(high.cls[j])]
                if not possible:continue
                choices,_,_=regions(planner.geom,row['current_crop'],np.asarray(row['mean'][:2]),high.xyxy,high.conf,high.cls,row['cls']);choices=choices[1:]
                if len(choices)!=2:continue
                transforms=[geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],high.xyxy[j],planner.g) for j in choices];valid=[(int(j),g) for j,g in zip(choices,transforms) if g['valid']]
                if len(valid)!=1:continue
                j,g=valid[0]
                if j not in possible:continue
                cv=dis_domain(frame_points(planner.g['points'],row['current_crop'],planner.g['eye_size']));rv=dis_domain(frame_points(planner.g['points'],row['history'][-1]['crop'],planner.g['eye_size']));pair=correspondence(g,row['history'],row['current_crop'],high.xyxy[j],planner.g,cv,rv)
                if pair is None:continue
                edges.append(dict(old_tid=row['tid'],new_tid=newcols[j],cost=float(context['C'][i,j]),row=i,column=j,raw_id=highids[j],box=high.xyxy[j].tolist(),row_data=row,transform=g,support=len(pair['unique']),age=frame-row['history'][-1]['frame']))
        if edges:events.append(dict(frame=frame,newly_confirmed=sorted(fresh),edges=edges))
        if len(out):flat.append(np.column_stack([np.full(len(out),frame),out]).astype(np.float32))
        offsets.append(offsets[-1]+len(out));seen.update(ids)
        for tid,t in current.items():
            if t.frame_id!=tracker.frame_id:continue
            planner.histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=frame,box=np.asarray(t.xyxy,np.float64).tolist(),crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))));planner.searches[tid]=[]
        if frame%200==0:status('native_val20_missing_metadata_recovery',video=video,frame=frame,total=total,events=len(events),guard=obs.read())
    if oldmat is not None:oldmat.close()
    dest.parent.mkdir(parents=True,exist_ok=True);npz=dest.with_suffix('.npz');metadata=dest.with_suffix('.json')
    values=np.concatenate(flat) if flat else np.empty((0,9),np.float32)
    with np.load(ROOT/entry['baseline'],allow_pickle=False) as z:
        assert int(z['frame_count'])==total and np.array_equal(values[:,:8],z['outputs']),(video,'Complete native baseline parity failed')
    checks['complete_native_baseline_outputs_exact']=len(values)
    with npz.open('xb') as f:np.savez_compressed(f,outputs=values,offsets=np.asarray(offsets,np.int64))
    write(metadata,dict(passed=True,video=video,frames=total,events=events,checks=dict(checks)),new=True)
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    d=dict(passed=True,video=video,frames=total,events=len(events),edges=sum(len(e['edges']) for e in events),checks=dict(checks),seconds=time.perf_counter()-began,artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [npz,metadata]});write(receipt,d,new=True);return d


def main(obs):
    protocol=read(RUN/'native_trace_protocol.json')
    for n,h in protocol['sources'].items():assert sha(ROOT/n)==h,n
    results={}
    for video in protocol['videos']:results[video]=trace(video,obs,protocol)
    for n,h in protocol['sources'].items():assert sha(ROOT/n)==h,n
    write(RUN/'native_trace_complete.json',dict(passed=True,videos=results,new_detector_calls=0,new_core_calls=0,scope='Missing native internal metadata recovered once. Existing query states and outputs exact; no altered tracker, old metrics or feature maps rerun.'),new=True);status('native_val20_trace_complete',videos=len(results),events=sum(r['events'] for r in results.values()))


if __name__=='__main__':main(guard())
