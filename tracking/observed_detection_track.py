"""Optional metadata-only XYWHByteTrack retaining real detector observations."""
from __future__ import annotations
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack,parse_bboxes
from tracking.xywh_bytetrack import XYWHByteTrack


class ObservedBOTrack(BOTrack):
    def update(self,new_track,frame_id):
        observed=new_track.observed_xyxy.copy()
        super().update(new_track,frame_id)
        self.observed_xyxy=observed;self.observed_frame=int(frame_id)-1

    def re_activate(self,new_track,frame_id,new_id=False):
        observed=new_track.observed_xyxy.copy()
        super().re_activate(new_track,frame_id,new_id=new_id)
        self.observed_xyxy=observed;self.observed_frame=int(frame_id)-1


class ObservationRecordingXYWHByteTrack(XYWHByteTrack):
    """Same assignments and Kalman state, plus separate raw-box metadata."""
    def init_track(self,results,img=None):
        if len(results)==0:return []
        tracks=[]
        for xywh,score,cls,raw in zip(parse_bboxes(results),results.conf,results.cls,results.xyxy):
            t=ObservedBOTrack(xywh,score,cls)
            t.observed_xyxy=np.asarray(raw,dtype=float).copy();t.observed_frame=int(self.frame_id)-1
            tracks.append(t)
        return tracks
