"""Causal full rollout of free-remainder recovery, with untouched later lifecycle."""
import gzip,time
import numpy as np
from research_singleton_free_20261108.common import *
from research_singleton_free_20261108.full_features import FULL,Representations
from research_singleton_free_20261108.full_planner import Planner
from research_identity_assignment_run_20261108.full_run import LiveTracker as ParentLive,scalabel
from research_identity_assignment_run_20261108.replay import Detections
from tracking.free_singleton_assignment import FreeSingletonAssociation,free_singleton_matches
from tracking.offline_association import make_tracker_args


class Tracker(FreeSingletonAssociation,ParentLive):
    def _first_association(self,pool,detections,activated,refind):
        self.pre=dict(frame=self.frame_id-1,rows=self.planner.rows(pool),cost=self.get_dists(pool,detections),
            classes=[d.cls for d in detections])
        return super()._first_association(pool,detections,activated,refind)


def rollout(video,arm,obs,backbone=None,*,zero=False):
    p=read(FULL/'protocol.json');entry=p['videos'][video];folder=FULL/'rollouts'/arm;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(video+'.receipt.json');rawpath=folder/(video+'.npz');jsonpath=FULL/'outputs'/arm/(video+'.json')
    if marker.exists():
        result=read(marker)
        for name,h in {**result['artifacts'],**result['feature_files']}.items():assert sha(ROOT/name)==h
        return result
    assert not rawpath.exists() and not jsonpath.exists()
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    clock=read(FULL/'runtime_inputs'/(video+'.json'))['anchors']
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order]
    settings=dict(alpha=0,beta=0) if zero else p['parameters'][arm];rep=None
    if settings['beta'] and arm!='U_SHIFT':
        kind={'U_R1':'R1','U_IMAGE':'IMAGE','U_DIS':'DIS'}.get(arm,'C');rep=Representations(video,kind,obs,backbone)
    planner=Planner(video,arm,rep);tracker=Tracker(make_tracker_args(**p['native_config']),planner,arm=arm,**settings)
    flat=[];rows=[];events=[];began=time.perf_counter()
    for frame in range(entry['frames']):
        lo,hi=np.searchsorted(frames,[frame,frame+1]);ids=order[lo:hi]
        out=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids))
        if len(out):flat.append(np.column_stack([np.full(len(out),frame),np.asarray(out)[:,:7]]).astype(np.float32))
        if str(frame) in clock:rows.append(scalabel(video,clock[str(frame)],out))
        event=tracker.identity_events[-1] if tracker.identity_events and tracker.identity_events[-1]['frame']==tracker.frame_id else None
        if event and event['eligible_free_edges']:
            pre=tracker.pre;snapshot=planner.last
            plan=snapshot['plan'] if snapshot is not None and snapshot['frame']==frame else dict(votes=[],columns={})
            C=pre['cost'];native=np.asarray(event['native_matches'],int).reshape(-1,2)
            rr=sorted(set(range(C.shape[0]))-set(native[:,0]));cc=sorted(set(range(C.shape[1]))-set(native[:,1]))
            matches,_,_,corrected,allowed=free_singleton_matches(C,native,rr,cc,plan,settings['alpha'],settings['beta'],arm,
                [r['state'] for r in pre['rows']],[r['cls'] for r in pre['rows']],pre['classes'])
            assert matches.tolist()==event['matches'] and set(map(tuple,native)).issubset(set(map(tuple,matches)))
            emitted={int(r[4]) for r in out}
            events.append(dict(frame=frame,rows=pre['rows'],plan=plan,C=C.tolist(),C_dtype=str(C.dtype),corrected=corrected.tolist(),
                classes=list(map(int,pre['classes'])),native_first=native.tolist(),actual_first=matches.tolist(),allowed_free_edges=[list(v) for v in allowed],
                emitted_ids=sorted(emitted),assigned={str(k):v for k,v in tracker.assigned.items() if k in emitted},duplicates=tracker.dupes))
        if frame%100==0:status('full_free_remainder_rollout',video=video,arm=arm,frame=frame,total=entry['frames'],events=len(events),guard=obs.read())
    values=np.concatenate(flat) if flat else np.zeros((0,8),np.float32)
    if zero:
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:assert np.array_equal(values,z['outputs']) and int(z['frame_count'])==entry['frames']
        assert rows==read(ROOT/entry['baseline'])
    with rawpath.open('xb') as f:np.savez_compressed(f,outputs=values,frame_count=entry['frames'])
    write(jsonpath,rows,new=True);eventpath=folder/(video+'.events.json.gz')
    with gzip.open(eventpath,'xt',encoding='utf-8') as f:json.dump(events,f,ensure_ascii=False,allow_nan=False)
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    result=dict(passed=True,video=video,arm=arm,zero=zero,frames=entry['frames'],seconds=time.perf_counter()-began,
        artifacts={path.relative_to(ROOT).as_posix():sha(path) for path in [rawpath,jsonpath,eventpath]},
        feature_files={} if rep is None else rep.files,reused_model_maps={} if rep is None else rep.reused,
        new_feature_maps=0 if rep is None else rep.new,core_imported=any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules))
    if arm not in ['U_FLYVIS','U_PERM']:assert not result['core_imported']
    write(marker,result,new=True);return result
