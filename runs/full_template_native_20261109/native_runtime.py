"""Causal verifier before global native assignment, with the native lifecycle.

No annotation interface. Reference/search histories reproduce the archived
Planner; final heads use exactly its seven original geometry features.
"""
from collections import deque,Counter
import hashlib,json
import numpy as np
from ultralytics.trackers.utils import matching,stracks as pool_helpers
from tracking.offline_association import _DetSlice
from tracking.spatial_identity_assignment import SpatialIdentityByteTrack,geometric_match
from tracking.foveated_eyes import crop_box
from tracking.z13_readout import HexGeometry
from tracking.z13_region_assignment import regions
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.channel_identity_head import causal_geometry,spatial_channel_features,decision_function
from common import ROOT,OUT,LEGACY,read,sha


def sequence_key(video,sequence):
    return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]


class Detections(_DetSlice):
    def __init__(self,b,s,c,ids):
        super().__init__(b,s,c);self.raw_ids=np.asarray(ids)
    def __getitem__(self,mask):
        return Detections(self._xyxy[mask],self.conf[mask],self.cls[mask],self.raw_ids[mask])


class FeatureBank:
    def __init__(self,video,arm,backbone=None):
        self.video=video;self.arm=arm;self.backbone=backbone
        self.cache={};self.files={};self.views=None;self.new=0;self.parity=[]
        self.folder=OUT/'features'/video;self.folder.mkdir(parents=True,exist_ok=True)
        self.norm={k:read(ROOT/f'runs/z13_target_center_dev_20261018/{k}/normalization.json') for k in ['R1','C']}
        # Closed old receipts, rather than filename alone, authorize cache reuse.
        self.legacy_sources={}
        for name in ['G_R1','G_FLYVIS']:
            path=LEGACY/'rollouts'/name/(video+'.receipt.json')
            if path.exists():
                receipt=read(path);self.legacy_sources.update(receipt.get('feature_files',{}))
                self.files[path.relative_to(ROOT).as_posix()]=sha(path)
        g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
        self.neighbors=HexGeometry(np.array(g['points']),g['eye_size']).neighbors

    def _load(self,path):
        name=path.relative_to(ROOT).as_posix();digest=sha(path)
        if path.is_relative_to(LEGACY):assert self.legacy_sources.get(name)==digest,name
        else:
            meta=read(path.with_suffix('.json'));assert meta['sha256']==digest
            assert meta['execution_sha256']==sha(OUT/'execution_protocol.json')
        self.files[name]=digest;return np.load(path,allow_pickle=False)

    def get(self,sequence,now):
        assert sequence and all(0<=int(f)<=now for f,_ in sequence)
        key=sequence_key(self.video,sequence)
        if key in self.cache:return self.cache[key]
        is_core=self.arm=='C34';slot='C34' if is_core else 'R1'
        path=self.folder/(key+'.'+slot+'.npy')
        legacy=LEGACY/'features'/self.video/(key+'.R1.npy')
        if path.exists():value=self._load(path)
        elif not is_core and legacy.exists():value=self._load(legacy)
        else:
            if self.views is None:
                from sequential_views import SequentialViews
                self.views=SequentialViews(self.video)
            hx=np.stack([self.views.get(f,c,now) for f,c in sequence]).astype(np.float32)
            kind='C' if is_core else 'R1';mean=np.array(self.norm[kind]['mean'])[:34];sd=np.array(self.norm[kind]['sd'])[:34]
            if is_core:
                import torch
                assert self.backbone is not None
                with torch.no_grad():
                    x=torch.from_numpy(hx[None]).to(self.backbone.dev);state=self.backbone.init(x[:,:1])
                    for j in range(1,len(hx)):state,feature=self.backbone.step(state,x[:,j-1],x[:,j])
                    value=((feature[0].cpu().numpy().T.astype(float)-mean)/sd).astype(np.float32)
                del x,state,feature
                old=LEGACY/'features'/self.video/(key+'.C.npy')
                if old.exists():
                    previous=self._load(old);error=float(np.max(abs(value[:,4:12]-previous)))
                    assert error<=1e-4,(key,error);self.parity.append(error)
            else:
                from scripts.z13_target_model import receptor_features_rich
                raw=receptor_features_rich(hx[None].astype(np.float64))[-1][0].T
                value=((raw-mean)/sd).astype(np.float32)
            assert value.shape==(721,34) and value.dtype==np.float32 and np.isfinite(value).all()
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            meta=dict(sha256=sha(path),execution_sha256=sha(OUT/'execution_protocol.json'),
                video=self.video,sequence=sequence,now=now,slot=slot,
                retinal_sources=dict(self.views.files),decoder_sources=dict(self.views.frames.sources))
            with path.with_suffix('.json').open('x',encoding='utf8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
            self.files[path.relative_to(ROOT).as_posix()]=meta['sha256'];self.new+=1
        assert value.shape==(721,34) and np.isfinite(value).all()
        if self.arm=='R134_spatial':
            from scripts.z13_target_model import neighbor_aug
            value=neighbor_aug(value.astype(float)[:,:17].T[None],self.neighbors)[0].T
        else:value=value.astype(float)
        if len(self.cache)>=500:self.cache.clear()
        self.cache[key]=value;return value

    def close(self):
        if self.views is not None:self.views.close()


def domain_reason(row,frame,native_unmatched):
    if row['cls']!=2:return 'not_car'
    if not native_unmatched:return 'native_first_already_matched'
    history=row['history']
    if not history or len({h['frame'] for h in history})<2:return 'insufficient_reference_history'
    gap=frame-history[-1]['frame']
    if gap not in (5,6):return 'outside_observed_training_gap'
    return None


class Planner:
    def __init__(self,video,arm,models=None,bank=None):
        self.video=video;self.arm=arm;self.models=models;self.bank=bank
        self.g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
        self.geom=HexGeometry(np.array(self.g['points'],np.float32),self.g['eye_size'])
        self.reader=ObservedSpatialReader(self.g['points'],self.g['eye_size'],[1280,720])
        axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
        self.offset_axes=axes@np.array([[13.,6.5],[0.,13.]])
        self.histories={};self.searches={};self.frame=-1;self.last=None;self.counts=Counter()

    def before(self,tracker,pool):
        self.frame=tracker.frame_id-1;self.last=None
        for t in pool:
            tid=int(t.track_id)
            if int(t.state)==2:self.searches.setdefault(tid,[]).append([self.frame,list(crop_box(tuple(t.mean[:4]),(1280,720)))])
            else:self.searches[tid]=[]

    def rows(self,pool):
        rows=[]
        for t in pool:
            tid=int(t.track_id);h=list(self.histories.get(tid,[]));h=[h[0]]*max(0,4-len(h))+h[-4:] if h else []
            rows.append(dict(tid=tid,state=int(t.state),cls=int(t.cls),start_frame=int(t.start_frame),
                mean=t.mean.tolist(),covariance=t.covariance.tolist(),history=h,
                current_crop=list(crop_box(tuple(t.mean[:4]),(1280,720))),search=list(self.searches.get(tid,[]))))
        return rows

    def logit(self,arm,geometry,channels=None):
        stats=self.models['geometry_normalization'];g=(np.array(geometry)-stats['mean'])/stats['scale']
        x=g if arm=='G7' else np.r_[g,channels]
        return float(decision_function(self.models['models'][arm],x[None])[0])

    def provide(self,tracker,pool,detections,cost):
        rows=self.rows(pool);assert len(detections)==len(self.high)
        native,unmatched,_=matching.linear_assignment(cost.copy(),thresh=tracker.args.match_thresh)
        native=np.asarray(native).reshape(-1,2).tolist();unmatched=set(map(int,unmatched))
        boxes=self.high.xyxy;scores=self.high.conf;classes=self.high.cls;votes=[];proposals=[]
        # Empty active path exercises the actual global solver but no proposals.
        if self.arm!='zero_empty':
            for i,row in enumerate(rows):
                if row['state']!=2 or not row['history']:continue
                choices,_,_=regions(self.geom,row['current_crop'],np.array(row['mean'][:2]),boxes,scores,classes,row['cls'])
                choices=choices[1:]
                if len(choices)!=2:continue
                eligible=[j for j in choices if geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],boxes[j],self.g)['valid']]
                if len(eligible)!=1:continue
                j=int(eligible[0]);h=row['history'];ref=h[-1]
                rec=dict(row=i,column=j,old_id=row['tid'],raw_id=int(self.high.raw_ids[j]),
                    original_cost=float(cost[i,j]),reference_frame=ref['frame'],reference_box=ref['box'],
                    reference_crop=ref['crop'],current_box=boxes[j].tolist(),current_crop=row['current_crop'],
                    candidate_columns=list(map(int,choices)),history=h,search=row['search'],accepted=False)
                proposals.append(rec);reason=domain_reason(row,self.frame,i in unmatched)
                if reason:rec['reason']=reason;self.counts[reason]+=1;continue
                rc=np.array(ref['crop']);ct=np.array(row['current_crop'])
                offsets=self.offset_axes*np.maximum((rc[2:]-rc[:2])/self.g['eye_size'],(ct[2:]-ct[:2])/self.g['eye_size'])
                args=(self.reader,ref['box'],rc,boxes[j],ct,offsets)
                old=relative_template_plan(*args,clip_to_candidate=True);full=relative_template_plan(*args)
                rec.update(old_cells=len(old['cells']),full_cells=len(full['cells']))
                if len(old['cells'])<3:
                    rec['reason']='insufficient_old_support';self.counts[rec['reason']]+=1;continue
                assert set(old['cells'])<=set(full['cells']) and len(full['cells'])>=3
                g=causal_geometry(ref['box'],boxes[j],cost[i,j],self.frame-ref['frame'],len(old['cells']))
                score=self.logit('G7',g);rec.update(geometry=g.tolist(),geometry_logit=score)
                if self.arm!='G7':
                    refseq=[[v['frame'],v['crop']] for v in h];currentseq=refseq+row['search']
                    assert currentseq[-1][0]==self.frame and max(f for f,_ in refseq)<self.frame
                    a=self.bank.get(refseq,self.frame);b=self.bank.get(currentseq,self.frame)
                    evidence=spatial_channel_features(self.reader.mesh,a,b,full['cells'],full['query'])
                    rec.update(evidence_valid=evidence['valid'],winner=evidence['winner'],
                        scores=evidence['scores'],channels=None if not evidence['valid'] else evidence['channels'].tolist())
                    if evidence['valid']:score=self.logit(self.arm,g,evidence['channels'])
                    else:rec['geometry_fallback']=True
                rec.update(logit=score,accepted=score>0,reason='head_accept' if score>0 else 'head_refuse')
                self.counts[rec['reason']]+=1
                if score>0:votes.append([i,j])
        plan=dict(votes=votes,columns={})
        self.last=dict(frame=self.frame,rows=rows,plan=plan,cost=cost.copy(),native_first=native,proposals=proposals)
        return plan

    def after(self,tracker):
        for t in tracker.tracked_stracks:
            if t.frame_id!=tracker.frame_id:continue
            tid=int(t.track_id)
            self.histories.setdefault(tid,deque(maxlen=4)).append(dict(frame=self.frame,
                box=np.asarray(t.xyxy,np.float64).tolist(),crop=list(crop_box(tuple(t.mean[:4]),(1280,720)))))
            self.searches[tid]=[]


class LiveTracker(SpatialIdentityByteTrack):
    def __init__(self,args,planner,**kwargs):
        super().__init__(args,provider=planner.provide,**kwargs);self.planner=planner;self.assigned={};self.dupes=[]
    def init_track(self,results,img=None):
        tracks=super().init_track(results,img)
        for t,rid in zip(tracks,results.raw_ids):t.export_raw_id=int(rid)
        return tracks
    def _apply_match(self,t,d,a,r):
        super()._apply_match(t,d,a,r);t.export_raw_id=d.export_raw_id;self.assigned[int(t.track_id)]=int(d.export_raw_id)
    def _pre_first_associate(self,pool,unconfirmed,img,high):
        super()._pre_first_associate(pool,unconfirmed,img,high);self.planner.high=high;self.planner.before(self,pool)
    def update(self,*args,**kwargs):
        self.assigned={};self.dupes=[];original=pool_helpers.remove_duplicate_stracks
        def dedup(first,second,dup_thresh=.15):
            distance=matching.iou_distance(first,second);aa,bb=original(first,second,dup_thresh)
            ka={x.track_id for x in aa};kb={x.track_id for x in bb}
            for i,j in zip(*np.where(distance<dup_thresh)):
                self.dupes.append(dict(tracked_id=int(first[i].track_id),lost_id=int(second[j].track_id),
                    tracked_duration=int(first[i].frame_id-first[i].start_frame),lost_duration=int(second[j].frame_id-second[j].start_frame),
                    iou=float(1-distance[i,j]),tracked_dropped=first[i].track_id not in ka,lost_dropped=second[j].track_id not in kb))
            return aa,bb
        pool_helpers.remove_duplicate_stracks=dedup
        try:value=super().update(*args,**kwargs)
        finally:pool_helpers.remove_duplicate_stracks=original
        self.planner.after(self);return value
