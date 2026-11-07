from types import SimpleNamespace
import numpy as np
from tracking.detection_supported_kalman import DetectionSupportedKalmanOutput


def test_disabled_is_exact_and_does_not_read_tracker():
    base=np.array([[0,0,10,10,1,.8,0,7]],dtype=np.float32)
    assert np.array_equal(DetectionSupportedKalmanOutput({0:'car'}).output(None,base,[],[],[]),base)


def test_enabled_reads_current_lost_state_without_mutating_it():
    track=SimpleNamespace(track_id=2,end_frame=9,is_activated=True,xyxy=np.array([0,0,10,10]),score=.6,cls=0)
    tracker=SimpleNamespace(frame_id=10,lost_stracks=[track]);before=track.xyxy.copy()
    out=DetectionSupportedKalmanOutput({0:'car'},enabled=True).output(tracker,np.zeros((0,8)),np.array([[0,0,10,10]]),[.2],[0])
    assert len(out)==1 and out[0,4]==2 and np.array_equal(track.xyxy,before) and track.end_frame==9


def test_expired_or_unsupported_lost_track_produces_nothing():
    t=SimpleNamespace(track_id=2,end_frame=1,is_activated=True,xyxy=np.array([0,0,10,10]),score=.6,cls=0)
    tracker=SimpleNamespace(frame_id=40,lost_stracks=[t])
    assert len(DetectionSupportedKalmanOutput({0:'car'},enabled=True).output(tracker,[],[[0,0,10,10]],[.2],[0]))==0
