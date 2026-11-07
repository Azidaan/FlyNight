"""Isolated query-memory write intervention, not the complete Deep OC-SORT."""
import numpy as np
from tracking.query_appearance_comparison import QueryAppearanceTracker
from tracking.lazy_appearance_association import Observation

def combine(previous,current,policy,score,threshold,beta=.9):
    current=np.asarray(current,np.float32)
    assert np.isfinite(current).all()
    if previous is None or policy=='last':return current.copy(),0.
    assert policy in ('mean','cautious')
    trust=float(np.clip((float(score)-threshold)/(1-threshold),0,1))
    alpha=beta if policy=='mean' else beta+(1-beta)*(1-trust)
    if alpha==1:return np.asarray(previous,np.float32).copy(),alpha
    value=alpha*np.asarray(previous,np.float32)+(1-alpha)*current
    norm=float(np.linalg.norm(value))
    assert norm>1e-12 and np.isfinite(value).all()
    return (value/norm).astype(np.float32),alpha

class MemoryWriteTracker(QueryAppearanceTracker):
    def __init__(self,args,server=None,weight=0.,shadow=False,policy='last',beta=.9):
        super().__init__(args,server,weight,shadow)
        assert policy in ('last','mean','cautious')
        self.policy=policy;self.beta=beta;self.write_events=[]
    def reset(self):
        super().reset();self.write_events.clear()
    def update_query_frame(self,detections,active):
        before=self.query_history.copy()
        result=super().update_query_frame(detections,active)
        if active and self.weight and self.policy!='last':
            for track in self.tracked_stracks:
                token=self.query_history.get(track.track_id)
                if token is None or token.frame!=self.frame_id-1:continue
                self.server.features([token])
                previous=before.get(track.track_id)
                vector,alpha=combine(None if previous is None else previous.vector,token.vector,
                    self.policy,track.score,self.args.track_high_thresh,self.beta)
                # Never mutate a shared real detection vector. Memory belongs to this tracker.
                memory=Observation(token.key,token.frame,token.crop.copy(),vector.copy())
                self.query_history[track.track_id]=memory
                self.write_events.append(dict(frame=token.frame,track_id=int(track.track_id),
                    previous_frame=None if previous is None else previous.frame,
                    score=float(track.score),alpha=alpha,vector=vector.tolist()))
        return result
