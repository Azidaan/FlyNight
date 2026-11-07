"""Appearance content versus a common cost offset, last-observation memory."""
import numpy as np
from tracking.query_appearance_comparison import QueryAppearanceTracker
from tracking.xywh_bytetrack import XYWHByteTrack

def patch_cost(original,appearance,eligible,mode,weight=.1):
    original=np.asarray(original);appearance=np.asarray(appearance,np.float32);eligible=np.asarray(eligible,bool)
    assert original.dtype==np.float32 and original.shape==appearance.shape==eligible.shape
    assert np.isfinite(original).all() and np.isfinite(appearance).all()
    assert np.all((appearance>=0)&(appearance<=1))
    if mode=='old':delta=np.float32(weight)*appearance
    elif mode=='centered':delta=np.float32(weight)*(appearance-np.float32(.5))
    elif mode=='shift':delta=np.full_like(original,-np.float32(weight*.5))
    else:raise ValueError(mode)
    return np.where(eligible,original+delta,original).astype(np.float32)

class CostUseTracker(QueryAppearanceTracker):
    def __init__(self,args,server=None,weight=0.,shadow=False,mode='centered'):
        # Compute the selected intervention before a real zero-shadow discards it.
        super().__init__(args,server,weight,False)
        assert mode in ('old','centered','shift')
        self.mode=mode;self.cost_shadow=shadow;self.write_events=[]
    def get_dists(self,tracks,detections):
        if not (self.query_active and self._inside_first and self.weight and len(tracks) and len(detections)):
            return super().get_dists(tracks,detections)
        original=XYWHByteTrack.get_dists(self,tracks,detections)
        eligible=original<=self.args.match_thresh
        ambiguous=np.zeros(len(tracks),bool)
        for i,row in enumerate(original):
            v=np.sort(row[eligible[i]])
            ambiguous[i]=len(v)>1 and v[1]-v[0]<=self.margin
        ambiguous|=np.any(eligible & ((eligible.sum(axis=0)>1)[None,:]),axis=1)
        ii=np.array([i for i in np.flatnonzero(ambiguous) if getattr(tracks[i],'observation',None) is not None],int)
        count=len(self.events);out=super().get_dists(tracks,detections)
        if not len(ii):
            assert len(self.events)==count
            return original if self.cost_shadow else out
        jj=np.flatnonzero(np.any(eligible[ii],axis=0));mask=eligible[np.ix_(ii,jj)]
        assert len(self.events)==count+1
        event=self.events[-1];part=original[np.ix_(ii,jj)]
        assert np.array_equal(part,np.asarray(event['original'],np.float32))
        assert event['track_ids']==[int(tracks[i].track_id) for i in ii]
        assert event['candidate_boxes']==[detections[j].xyxy.tolist() for j in jj]
        assert np.array_equal(mask,np.asarray(event['eligible'],bool))
        if self.mode!='old':
            updated=patch_cost(part,np.asarray(event['appearance'],np.float32),mask,self.mode,self.weight)
            out[np.ix_(ii,jj)]=updated;event['updated']=updated.tolist()
        if self.cost_shadow:
            self.shadow_pairs+=int(np.count_nonzero(out!=original))
            return original
        return out
    def update_query_frame(self,detections,active):
        before=self.query_history.copy();out=super().update_query_frame(detections,active)
        if active and self.weight:
            for track in self.tracked_stracks:
                token=self.query_history.get(track.track_id)
                if token is None or token.frame!=self.frame_id-1:continue
                previous=before.get(track.track_id)
                self.write_events.append(dict(frame=token.frame,track_id=int(track.track_id),query_key=list(token.key),
                    previous_frame=None if previous is None else previous.frame,accepted=True))
        return out
