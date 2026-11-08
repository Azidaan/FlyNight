"""Full causal tracker: native matches fixed, own-template veto only on free edges."""
import gzip,time
import numpy as np
from ultralytics.trackers.utils import matching
from tracking.free_singleton_assignment import FreeSingletonAssociation
from tracking.own_template_verification import correspondence,absolute_score,passes
from tracking.spatial_identity_assignment import geometric_match,frame_points
from tracking.z13_region_assignment import regions
from tracking.offline_association import make_tracker_args
from research_identity_assignment_run_20261108.full_planner import Planner as OriginalPlanner,dis_domain
from research_identity_assignment_run_20261108.full_run import LiveTracker,scalabel
from research_identity_assignment_run_20261108.replay import Detections
from research_singleton_verify_full_20261108.common import *
paths,feature_module=configure_paths()


class Planner(OriginalPlanner):
    def __init__(self,video,kind,threshold,representations=None):
        super().__init__(video,'G',representations);self.kind=kind;self.threshold=threshold

    def own_value(self,row,box,frame,column,transform):
        current_valid=dis_domain(frame_points(self.g['points'],row['current_crop'],self.g['eye_size']))
        reference_valid=dis_domain(frame_points(self.g['points'],row['history'][-1]['crop'],self.g['eye_size']))
        pair=correspondence(transform,row['history'],row['current_crop'],box,self.g,current_valid,reference_valid)
        if pair is None:return None,0
        if self.kind in ['SUPPORT','RANDOM']:current=reference=np.empty((721,0))
        else:
            ref=[[h['frame'],h['crop']] for h in row['history']];current=self.representations.get(ref+row['search'],frame);reference=self.representations.get(ref,frame)
        value=absolute_score(self.kind,current,reference,pair,transform['scale'],video=self.video,tid=row['tid'],
            last_frame=row['history'][-1]['frame'],frame=frame,column=column)
        return value,len(pair['unique'])

    def provide(self,tracker,pool,detections,cost):
        rows=self.rows(pool);boxes=self.high.xyxy;confidence=self.high.conf;classes=self.high.cls;decisions=[];votes=[]
        if self.kind=='GEO':
            plan=super().build(rows,boxes,confidence,classes,cost,self.frame,score=False)
        elif self.threshold=='off':plan=dict(votes=[],columns={})
        else:
            _,rr,cc=matching.linear_assignment(cost,thresh=.8);free_columns=set(map(int,cc))
            for raw_i in rr:
                i=int(raw_i);row=rows[i]
                if row['state']!=2 or not row['history']:continue
                choices,_,_=regions(self.geom,row['current_crop'],np.array(row['mean'][:2]),boxes,confidence,classes,row['cls']);choices=choices[1:]
                if len(choices)!=2:continue
                transforms=[geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],boxes[j],self.g) for j in choices]
                valid=[(int(j),g) for j,g in zip(choices,transforms) if g['valid']]
                if len(valid)!=1:continue
                j,transform=valid[0]
                if j not in free_columns or cost[i,j]>=1 or row['cls']!=int(classes[j]):continue
                value,support=self.own_value(row,boxes[j],self.frame,j,transform);accepted=passes(value,self.threshold)
                decisions.append(dict(row=i,column=j,score=value,support=support,accepted=accepted,threshold=self.threshold,
                    transform={k:v for k,v in transform.items() if k not in ['p','q','reference_ids']}))
                if accepted:votes.append([i,j])
            plan=dict(votes=votes,columns={})
        self.last=dict(frame=self.frame,rows=rows,plan=plan,decisions=decisions,boxes=np.asarray(boxes).tolist(),classes=np.asarray(classes).astype(int).tolist())
        return plan


class Tracker(FreeSingletonAssociation,LiveTracker):
    def _first_association(self,pool,detections,activated,refind):
        self.pre=dict(rows=self.planner.rows(pool),C=self.get_dists(pool,detections),classes=[int(d.cls) for d in detections])
        return super()._first_association(pool,detections,activated,refind)


def rollout(video,arm,obs,backbone=None,*,zero=False,disabled=False):
    p=read(FULL/'protocol.json');entry=p['videos'][video];folder=FULL/'rollouts'/arm;folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(video+'.receipt.json');outpath=FULL/'outputs'/arm/(video+'.json');rawpath=folder/(video+'.npz')
    if marker.exists():
        receipt=read(marker)
        for n,h in {**receipt['artifacts'],**receipt['feature_files']}.items():assert sha(ROOT/n)==h
        return receipt
    assert not outpath.exists() and not rawpath.exists()
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    clock=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];settings=p['parameters'].get(arm,dict(alpha=0,threshold='off'))
    threshold='off' if disabled or zero else settings['threshold'];kind='PERM' if disabled else arm
    alpha=0 if zero else settings.get('alpha',.2)
    if disabled:alpha=.2
    rep=None
    if kind in ['IMAGE','R1','DIS','FLYVIS','PERM'] and threshold!='off':rep=feature_module.Representations(video,'C' if kind in ['FLYVIS','PERM'] else kind,obs,backbone)
    planner=Planner(video,kind,threshold,rep);tracker=Tracker(make_tracker_args(**p['native_config']),planner,alpha=alpha,beta=0,arm='U_THRESHOLD' if kind=='THRESHOLD' else 'U_GEO')
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order];flat=[];output=[];events=[];began=time.perf_counter()
    for frame in range(entry['frames']):
        lo,hi=np.searchsorted(frames,[frame,frame+1]);ids=order[lo:hi]
        out=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids))
        if len(out):flat.append(np.column_stack([np.full(len(out),frame),np.asarray(out)[:,:7]]).astype(np.float32))
        if str(frame) in clock:output.append(scalabel(video,clock[str(frame)],out))
        e=tracker.identity_events[-1] if tracker.identity_events and tracker.identity_events[-1]['frame']==tracker.frame_id else None
        last=planner.last if planner.last is not None and planner.last['frame']==frame else None
        if e and (e['eligible_free_edges'] or last is not None and last['decisions']):
            C=tracker.pre['C'];emitted={int(r[4]) for r in out}
            events.append(dict(frame=frame,kind=kind,threshold=threshold,rows=tracker.pre['rows'],C=C.tolist(),C_dtype=str(C.dtype),
                classes=tracker.pre['classes'],boxes=None if last is None else last['boxes'],plan=dict(votes=[],columns={}) if last is None else last['plan'],
                decisions=[] if last is None else last['decisions'],native_first=e['native_matches'],actual_first=e['matches'],
                allowed_free_edges=e['eligible_free_edges'],emitted_ids=sorted(emitted),assigned={str(k):v for k,v in tracker.assigned.items() if k in emitted},duplicates=tracker.dupes))
        if frame%100==0:status('own_template_full_rollout',video=video,arm=arm,frame=frame,total=entry['frames'],events=len(events),guard=obs.read())
    values=np.concatenate(flat) if flat else np.zeros((0,8),np.float32)
    if zero or disabled:
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:assert np.array_equal(values,z['outputs']) and int(z['frame_count'])==entry['frames']
        assert output==read(ROOT/entry['baseline'])
    with rawpath.open('xb') as f:np.savez_compressed(f,outputs=values,frame_count=entry['frames'])
    write(outpath,output,new=True);eventpath=folder/(video+'.events.json.gz')
    with gzip.open(eventpath,'xt',encoding='utf-8') as f:json.dump(events,f,ensure_ascii=False,allow_nan=False)
    for n,h in entry['sources'].items():assert sha(ROOT/n)==h
    receipt=dict(passed=True,video=video,arm=arm,zero=zero,disabled_gate_check=disabled,frames=entry['frames'],seconds=time.perf_counter()-began,
        artifacts={v.relative_to(ROOT).as_posix():sha(v) for v in [rawpath,outpath,eventpath]},feature_files={} if rep is None else rep.files,
        reused_model_maps={} if rep is None else rep.reused,new_feature_maps=0 if rep is None else rep.new,
        core_imported=any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules))
    if kind not in ['FLYVIS','PERM']:assert not receipt['core_imported']
    write(marker,receipt,new=True);return receipt
