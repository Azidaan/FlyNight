"""Do not indefinitely retain a write-rejected appearance observation.

The forced-refresh age is the unchanged track lifetime, not a fitted threshold.
"""
from tracking.query_assignment_write import AssignmentWriteTracker
from tracking.query_appearance_comparison import QueryAppearanceTracker

class BoundedWriteTracker(AssignmentWriteTracker):
    def update_query_frame(self,detections,active):
        result=super().update_query_frame(detections,active)
        if active and self.weight:
            for event in reversed(self.write_events):
                if event['frame']!=self.frame_id-1:break
                if not event['accepted'] and event['previous_frame'] is not None and event['frame']-event['previous_frame']>=self.max_frames_lost:
                    token=self.server.tokens[tuple(event['query_key'])]
                    assert token.frame==self.frame_id-1
                    self.query_history[event['track_id']]=token
                    event.update(accepted=True,reason='forced_refresh_at_original_track_lifetime')
        return result

class PeriodicWriteTracker(QueryAppearanceTracker):
    def __init__(self,args,server=None,weight=0.,shadow=False):
        super().__init__(args,server,weight,shadow);self.write_events=[]
    def reset(self):
        super().reset();self.write_events.clear()
    def update_query_frame(self,detections,active):
        before=self.query_history.copy();result=super().update_query_frame(detections,active)
        if active and self.weight:
            for track in self.tracked_stracks:
                token=self.query_history.get(track.track_id)
                if token is None or token.frame!=self.frame_id-1:continue
                old=before.get(track.track_id)
                accepted=old is None or token.frame-old.frame>=self.max_frames_lost
                if not accepted:self.query_history[track.track_id]=old
                self.write_events.append(dict(frame=token.frame,track_id=int(track.track_id),query_key=list(token.key),
                    previous_frame=None if old is None else old.frame,accepted=accepted,
                    reason='initialization' if old is None else ('periodic_refresh_at_original_track_lifetime' if accepted else 'period_not_elapsed')))
        return result
