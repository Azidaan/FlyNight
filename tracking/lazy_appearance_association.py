"""Причинная проверка внешности спорных назначений внутри XYWHByteTrack."""
from dataclasses import dataclass,field
import hashlib
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack,parse_bboxes
from ultralytics.trackers.basetrack import BaseTrack
from tracking.xywh_bytetrack import XYWHByteTrack

@dataclass
class Observation:
    key:tuple
    frame:int
    crop:object
    vector:object=None
    nulls:dict=field(default_factory=dict)

class AppearanceServer:
    """Общие неизменные признаки реальных детекций; история назначений раздельна."""
    def __init__(self,encoder=None):
        self.encoder=encoder;self.frame=-1;self.image=None;self.tokens={};self.embedded=0
    def reset(self):
        """Start a new video; call once on the shared server, outside trackers."""
        self.frame=-1;self.image=None;self.tokens.clear();self.embedded=0
    def start_frame(self,frame,image):
        assert frame>self.frame;self.frame=frame;self.image=image
        self.tokens={k:t for k,t in self.tokens.items() if t.frame>=frame-32}
    def observation(self,box):
        from tracking.recovery_admission_features import crop
        b=np.asarray(box,np.float32);key=(self.frame,*b.tolist())
        if key not in self.tokens:self.tokens[key]=Observation(key,self.frame,crop(self.image,b))
        return self.tokens[key]
    def features(self,tokens,null_seed=None):
        assert all(t.frame<=self.frame for t in tokens)
        if null_seed is not None:
            for t in tokens:
                if null_seed not in t.nulls:
                    seed=int.from_bytes(hashlib.sha256((str(t.key)+'/'+str(null_seed)).encode()).digest()[:8],'little')
                    z=np.random.default_rng(seed).normal(size=512).astype(np.float32);t.nulls[null_seed]=z/np.linalg.norm(z)
            return np.stack([t.nulls[null_seed] for t in tokens])
        missing=list({id(t):t for t in tokens if t.vector is None}.values())
        if missing:
            assert self.encoder is not None
            import torch
            with torch.inference_mode():
                for start in range(0,len(missing),16):
                    group=missing[start:start+16];a=np.stack([t.crop for t in group]).astype(np.float32)/255
                    a=(a-np.array([.406,.456,.485],np.float32))/np.array([.225,.224,.299],np.float32)
                    x=torch.from_numpy(a[...,::-1].copy()).permute(0,3,1,2).to('cuda');_,z=self.encoder(x)
                    z=torch.nn.functional.normalize(z,dim=1).cpu().numpy();assert np.isfinite(z).all()
                    for t,v in zip(group,z):t.vector=v
                    self.embedded+=len(group)
        return np.stack([t.vector for t in tokens])

class ObservationTrack(BOTrack):
    def update(self,new_track,frame_id):
        super().update(new_track,frame_id);self.observation=new_track.observation
    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id);self.observation=new_track.observation

class LazyAppearanceXYWH(XYWHByteTrack):
    def __init__(self,args,server=None,weight=0.,null_seed=None,margin=.15):
        external=BaseTrack._count
        try:super().__init__(args)
        finally:BaseTrack._count=external
        self.server=server;self.weight=float(weight);self.null_seed=null_seed;self.margin=margin
        self._id_count=0;self._inside_first=False;self.events=[];self.changed_cost_frames=0
    def reset(self):
        """Reset local video state without touching another variant's ID counter."""
        external=BaseTrack._count
        try:super().reset()
        finally:BaseTrack._count=external
        self._id_count=0;self._inside_first=False
        self.events.clear();self.changed_cost_frames=0
    def update(self,*args,**kwargs):
        external=BaseTrack._count;BaseTrack._count=self._id_count
        try:return super().update(*args,**kwargs)
        finally:self._id_count=BaseTrack._count;BaseTrack._count=external
    def init_track(self,results,img=None):
        if len(results)==0:return []
        tracks=[ObservationTrack(b,s,c) for b,s,c in zip(parse_bboxes(results),results.conf,results.cls)]
        for t in tracks:t.observation=None if self.server is None or not self.weight else self.server.observation(t.xyxy)
        return tracks
    def _first_association(self,*args):
        self._inside_first=True
        try:return super()._first_association(*args)
        finally:self._inside_first=False
    def get_dists(self,tracks,detections):
        original=super().get_dists(tracks,detections)
        if not self._inside_first or self.weight==0 or not len(tracks) or not len(detections):return original
        eligible=original<=self.args.match_thresh
        # Только спорные исходные кандидаты; новых геометрических рёбер нет.
        ambiguous=np.zeros(len(tracks),bool)
        for i,row in enumerate(original):
            values=np.sort(row[eligible[i]])
            ambiguous[i]=len(values)>1 and values[1]-values[0]<=self.margin
        ambiguous |= np.any(eligible & ((eligible.sum(axis=0)>1)[None,:]),axis=1)
        ii=np.array([i for i in np.flatnonzero(ambiguous) if getattr(tracks[i],'observation',None) is not None],int)
        if not len(ii):return original
        jj=np.flatnonzero(np.any(eligible[ii],axis=0));assert len(jj)
        previous=[tracks[i].observation for i in ii];current=[detections[j].observation for j in jj]
        assert all(t.frame<self.frame_id-1 for t in previous) and all(t.frame==self.frame_id-1 for t in current)
        vectors=self.server.features(previous+current,self.null_seed);a=vectors[:len(ii)];b=vectors[len(ii):]
        appearance=np.clip((1-a@b.T)/2,0,1);out=original.copy();mask=eligible[np.ix_(ii,jj)]
        out[np.ix_(ii,jj)]=np.where(mask,original[np.ix_(ii,jj)]+self.weight*appearance,original[np.ix_(ii,jj)])
        self.changed_cost_frames+=1
        self.events.append(dict(frame=self.frame_id-1,track_ids=[int(tracks[i].track_id) for i in ii],past_frames=[t.frame for t in previous],
            candidate_boxes=[detections[j].xyxy.tolist() for j in jj],original=original[np.ix_(ii,jj)].tolist(),updated=out[np.ix_(ii,jj)].tolist(),eligible=mask.tolist(),
            appearance=appearance.tolist(),null_seed=self.null_seed))
        return out
