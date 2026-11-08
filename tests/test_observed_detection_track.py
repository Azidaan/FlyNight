import numpy as np
import pytest
from ultralytics.trackers.basetrack import BaseTrack
from tracking.offline_association import _DetSlice,make_tracker_args
from tracking.xywh_bytetrack import XYWHByteTrack
from tracking.observed_detection_track import ObservationRecordingXYWHByteTrack


@pytest.mark.parametrize('fuse',[True,False])
def test_recording_preserves_native_outputs_and_updates_raw_observations_in_all_stages(fuse):
    frames=[([[20,20,40,40]],[.9]),([[22,20,42,40]],[.9]),([[25,20,45,40]],[.2]),([],[]),([[28,20,48,40]],[.9]),([[30,20,50,40],[70,20,90,40]],[.9,.8]),([[32,20,52,40],[72,20,92,40]],[.9,.9])]
    outputs=[]
    for cls in [XYWHByteTrack,ObservationRecordingXYWHByteTrack]:
        BaseTrack.reset_id();tracker=cls(make_tracker_args(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=fuse));values=[]
        for frame,(boxes,conf) in enumerate(frames):
            b=np.asarray(boxes,np.float32).reshape(-1,4);s=np.asarray(conf,np.float32);c=np.full(len(b),2,np.float32)
            out=tracker.update(_DetSlice(b,s,c));values.append(out.copy())
            if cls is ObservationRecordingXYWHByteTrack:
                for t in tracker.tracked_stracks:
                    if t.frame_id!=tracker.frame_id:continue
                    assert t.observed_frame==frame
                    assert any(np.array_equal(t.observed_xyxy,box.astype(float)) for box in b)
                if frame==1:
                    t=tracker.tracked_stracks[0]
                    assert np.array_equal(t.observed_xyxy,b[0].astype(float)) and not np.array_equal(t.xyxy,t.observed_xyxy)
        outputs.append(values)
    assert all(np.array_equal(a,b) for a,b in zip(*outputs))
