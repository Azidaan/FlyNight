"""Recover missing rival histories and full native lifecycle on archived queries."""
import os,sys,time,pickle,zipfile
from collections import deque,defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def sequence_key(video,sequence):
    return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]


def export(video,queries):
    import numpy as np
    from ultralytics.trackers.basetrack import BaseTrack,TrackState
    from ultralytics.trackers.utils import matching
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking.foveated_eyes import crop_box
    from tracking.z13_readout import HexGeometry
    from tracking.z13_region_assignment import regions
    from tracking.spatial_identity_assignment import geometric_match
    folder=RUN/'exports';folder.mkdir(exist_ok=True);dest=folder/(video+'.json')
    if dest.exists():
        result=read(dest)
        for n,h in result['sources'].items():assert sha(ROOT/n)==h,n
        assert sha(dest.with_suffix('.npz'))==result['matrices_sha256']
        assert sha(folder/(video+'.states.zip'))==result['states_sha256']
        return result
    began=time.perf_counter();raw=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
    with np.load(raw,allow_pickle=False) as z:b=z['xyxy'];s=z['conf'];cl=z['cls'];frames=z['frame_idx']
    meta_path=ROOT/f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json'
    meta=read(meta_path)['rows'];by_pair={(r['current_frame'],r['tid']):r for r in meta}
    wanted=defaultdict(list)
    for q in queries:wanted[q['frame']].append(q)
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    geom=HexGeometry(np.array(geometry['points'],np.float32),geometry['eye_size'])
    histories={};searches={};snapshots={};matrices={};requests={};assignments={};context={'frame':-1}
    original_path=ROOT/f'runs/spatial_identity_maps_20261107/tracker/{video}.json'
    original=read(original_path) if original_path.exists() else None
    original_rows={r['uid']:r for r in original['rows']} if original else {}
    archived_matrices=np.load(original_path.with_suffix('.npz'),allow_pickle=False) if original else None

    class Detections(_DetSlice):
        def __init__(self,boxes,scores,classes,ids):super().__init__(boxes,scores,classes);self.raw_ids=np.asarray(ids)
        def __getitem__(self,mask):return Detections(self._xyxy[mask],self.conf[mask],self.cls[mask],self.raw_ids[mask])

    class Tracker(XYWHByteTrack):
        def init_track(self,results,img=None):
            tracks=super().init_track(results,img)
            for t,raw_id in zip(tracks,results.raw_ids):t.export_raw_id=int(raw_id)
            return tracks

        def _pre_first_associate(self,pool,unconfirmed,img,high):
            super()._pre_first_associate(pool,unconfirmed,img,high)
            f=context['frame'];context['high_ids']=high.raw_ids.tolist()
            for t in pool:
                tid=int(t.track_id)
                if t.state==TrackState.Lost:searches.setdefault(tid,[]).append([f,list(crop_box(tuple(t.mean[:4]),(1280,720)))])
                else:searches[tid]=[]
            if f not in wanted:return
            context['pool']=pool;context['high']=high

        def _first_association(self,pool,detections,activated,refind):
            f=context['frame']
            if f not in wanted:return super()._first_association(pool,detections,activated,refind)
            cost=self.get_dists(pool,detections);high=context['high'];rows=[];votes=[];proposal_rows={}
            for i,t in enumerate(pool):
                tid=int(t.track_id);h=list(histories.get(tid,[]));h=[h[0]]*max(0,4-len(h))+h[-4:] if h else []
                crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))
                row=dict(tid=tid,state=int(t.state),cls=int(t.cls),start_frame=int(t.start_frame),
                    mean=t.mean.tolist(),covariance=t.covariance.tolist(),history=h,current_crop=crop,
                    current_uid=by_pair.get((f,tid),{}).get('uid'),search=list(searches.get(tid,[])))
                rows.append(row)
                if t.state!=TrackState.Lost or not h:continue
                choices,_,_=regions(geom,crop,t.mean[:2],high.xyxy,high.conf,high.cls,t.cls)
                choices=choices[1:];row['choices']=choices
                m=by_pair.get((f,tid))
                if m is not None:
                    assert choices==m['current_choices'] and crop==m['current_crop'],(video,f,tid,'region parity')
                if len(choices)!=2:continue
                gs=[geometric_match(t.mean,t.covariance,h,crop,high.xyxy[j],geometry) for j in choices]
                row['geometry']=[{k:v for k,v in g.items() if k not in ['p','q','reference_ids']} for g in gs]
                eligible=[j for j,g in zip(choices,gs) if g['valid']]
                if len(eligible)==1:votes.append([i,eligible[0]]);proposal_rows.setdefault(eligible[0],[]).append(i)
            columns={}
            def request(row,sequence,uid=None):
                key=sequence_key(video,sequence)
                entry=dict(key=key,video=video,sequence=sequence,uid=uid)
                if key in requests and requests[key]['uid'] is None and uid is not None:requests[key]['uid']=uid
                else:requests.setdefault(key,entry)
                return key
            for j,seeds in proposal_rows.items():
                rivals=sorted(set(np.flatnonzero(cost[:,j]<=.8).tolist()+seeds))
                observer=min(seeds,key=lambda i:(float(np.trace(np.array(rows[i]['covariance'])[:2,:2])),rows[i]['start_frame'],rows[i]['tid']))
                view=rows[observer];record=dict(rows=rivals,geometry_valid=True,observer_tid=view['tid'],current_crop=view['current_crop'])
                if len(rivals)<2 or any(rows[i]['cls']!=int(detections[j].cls) or not rows[i]['history'] for i in rivals):
                    record['geometry_valid']=False;columns[str(j)]=record;continue
                prefix=[[h['frame'],h['crop']] for h in view['history']];sequence=prefix+view['search']
                record['current_key']=request(view,sequence,view['current_uid']);items=[]
                for i in rivals:
                    row=rows[i];g=geometric_match(row['mean'],row['covariance'],row['history'],view['current_crop'],high.xyxy[j],geometry)
                    if not g['valid'] or len(set(g.get('q',[])))<3:record['geometry_valid']=False
                    refseq=[[h['frame'],h['crop']] for h in row['history']]
                    items.append(dict(row=i,reference_key=request(row,refseq),mapping=g))
                record['items']=items;columns[str(j)]=record
            matches,ut,ud=matching.linear_assignment(cost,thresh=self.args.match_thresh)
            matrix_key='cost_'+str(f);matrices[matrix_key]=cost.copy()
            if original and str(f) in original['snapshots']:
                om=original['snapshots'][str(f)]
                assert om['track_ids']==[r['tid'] for r in rows]
                assert np.array_equal(cost,archived_matrices[matrix_key])
                assert np.array_equal(np.asarray(matches),np.asarray(om['matches']))
            snapshots[str(f)]=dict(frame=f,rows=rows,raw_ids=high.raw_ids.tolist(),boxes=high.xyxy.tolist(),
                confidence=high.conf.tolist(),classes=high.cls.tolist(),votes=votes,columns=columns,
                native_matches=np.asarray(matches).reshape(-1,2).tolist(),queries=wanted[f])
            for q in wanted[f]:
                m=by_pair[(f,q['tid'])];i=next(i for i,r in enumerate(rows) if r['tid']==q['tid'])
                assert rows[i].get('choices')==q['candidate_high_ids']
                if q['uid'] in original_rows:
                    old=original_rows[q['uid']]
                    assert np.array_equal(rows[i]['mean'],old['mean']) and np.array_equal(rows[i]['covariance'],old['covariance'])
                    assert rows[i]['history']==old['history']
            self._apply_matches(matches,pool,detections,activated,refind)
            return ut,ud

        def _apply_match(self,t,det,activated,refind):
            result=super()._apply_match(t,det,activated,refind)
            t.export_raw_id=det.export_raw_id
            if context['frame'] in wanted:assignments[(context['frame'],int(t.track_id))]=int(det.export_raw_id)
            return result

    BaseTrack.reset_id();cfg=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config']
    tracker=Tracker(make_tracker_args(**cfg));order=np.argsort(frames,kind='stable');sorted_frames=frames[order]
    state_path=folder/(video+'.states.zip');state_file=zipfile.ZipFile(state_path,'x',compression=zipfile.ZIP_DEFLATED)
    for f in range(max(wanted)+1):
        context['frame']=f;lo,hi=np.searchsorted(sorted_frames,[f,f+1]);ix=order[lo:hi]
        if f in wanted:
            state=dict(frame_id=tracker.frame_id,count=BaseTrack._count,
                pools={n:[t.__dict__ for t in getattr(tracker,n)] for n in ['tracked_stracks','lost_stracks','removed_stracks']})
            state_file.writestr(str(f)+'.pkl',pickle.dumps(state,protocol=5))
        output=tracker.update(Detections(b[ix],s[ix],cl[ix],ix))
        highids=context['high_ids'];final={}
        for t in tracker.tracked_stracks:
            if t.frame_id!=tracker.frame_id:continue
            tid=int(t.track_id)
            if f in wanted:
                raw_id=assignments.get((f,tid))
                if raw_id is None:
                    idx=int(t.idx);raw_id=int(highids[idx])
                final[str(tid)]=raw_id
            histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=f,box=np.asarray(t.xyxy,np.float64).tolist(),crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))))
            searches[tid]=[]
        if f in wanted:
            snap=snapshots[str(f)];snap.update(native_final_assignments=final,native_output=output.tolist(),
                first_assignments={str(tid):raw_id for (ff,tid),raw_id in assignments.items() if ff==f})
    state_file.close()
    if archived_matrices is not None:archived_matrices.close()
    npz=dest.with_suffix('.npz')
    with npz.open('xb') as f:np.savez_compressed(f,**matrices)
    sources={n:sha(ROOT/n) for n in [str(raw.relative_to(ROOT)),str(meta_path.relative_to(ROOT))]}
    result=dict(passed=True,video=video,snapshots=snapshots,requests=requests,query_count=len(queries),
        sources=sources,matrices_sha256=sha(npz),states_sha256=sha(state_path),seconds=time.perf_counter()-began)
    write(dest,result,new=True);return result


def main(obs):
    assert read(RUN/'zero_parity.json')['passed'];verify_phase('export_phase.json')
    queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']
    first=read(RUN/'zero_phase.json')['first_uid'];old=next(r for r in read(ROOT/'research_spatial_identity_20261107/readiness_rows.json') if r['uid']==first)
    engineer=dict(uid=first,video=old['video'],split='fit84',frame=old['current_frame'],tid=0,episode=old['episode'],candidate_high_ids=old['detector_candidates'])
    m=next(r for r in read(ROOT/f"runs/z13_temporal_pair_readiness_20261025/inputs/{old['video']}.json")['rows'] if r['uid']==first);engineer['tid']=m['tid']
    groups=defaultdict(list)
    for q in [engineer]+queries:groups[q['video']].append(q)
    order=[engineer['video']]+sorted(set(groups)-{engineer['video']});manifest={}
    for n,video in enumerate(order):
        result=export(video,groups[video]);manifest[video]=sha(RUN/'exports'/(video+'.json'))
        status('native_state_export',complete=n+1,total=len(order),video=video,requests=len(result['requests']),guard=obs.read())
    verify_phase('export_phase.json')
    write(RUN/'export_complete.json',dict(passed=True,videos=manifest,queries=len(queries),engineer_uid=first),new=True)


if __name__=='__main__':main(guard())
