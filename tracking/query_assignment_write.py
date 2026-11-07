"""Separate assignment from write admission; a KeepTrack-inspired component only."""
import numpy as np
from tracking.query_appearance_comparison import QueryAppearanceTracker
from tracking.xywh_bytetrack import XYWHByteTrack

def safe_write_mask(cost,threshold):
    eligible=np.asarray(cost)<=threshold
    return eligible & (eligible.sum(axis=1)==1)[:,None] & (eligible.sum(axis=0)==1)[None,:]

class AssignmentWriteTracker(QueryAppearanceTracker):
    def __init__(self,args,server=None,weight=0.,shadow=False):
        super().__init__(args,server,weight,shadow)
        self.write_events=[];self.safe_keys=set()
    def reset(self):
        super().reset();self.write_events.clear();self.safe_keys.clear()
    def get_dists(self,tracks,detections):
        if self.query_active and self._inside_first and len(tracks) and len(detections):
            original=XYWHByteTrack.get_dists(self,tracks,detections)
            mask=safe_write_mask(original,self.args.match_thresh)
            for i,j in np.argwhere(mask):
                token=getattr(detections[j],'observation',None)
                if token is not None:self.safe_keys.add((tracks[i].track_id,token.key))
        return super().get_dists(tracks,detections)
    def update_query_frame(self,detections,active):
        previous=self.query_history.copy();self.safe_keys=set()
        result=super().update_query_frame(detections,active)
        if active and self.weight:
            for track in self.tracked_stracks:
                current=self.query_history.get(track.track_id)
                if current is None or current.frame!=self.frame_id-1:continue
                old=previous.get(track.track_id)
                accepted=old is None or (track.track_id,current.key) in self.safe_keys
                if not accepted:self.query_history[track.track_id]=old
                self.write_events.append(dict(frame=current.frame,track_id=int(track.track_id),
                    query_key=list(current.key),previous_frame=None if old is None else old.frame,
                    accepted=accepted,reason='initialization' if old is None else ('unique_first_pair' if accepted else 'ambiguous_or_not_first_association')))
        return result
