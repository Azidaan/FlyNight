"""A new video must not inherit old optical flow or birth context."""
import numpy as np
import pytest

from tracking.birth_policy import BirthHead, QBirthByteTrack, DeferredBirthByteTrack
from tracking.decision_log import IndexedDetections
from tracking.flyvis_tracker import MotionField
from tracking.motion_augmented_bytetrack import MotionAugmentedByteTrack
from tracking.motion_source_bytetrack import MotionSourceByteTrack, make_deferred_cascade
from tracking.offline_association import make_tracker_args
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


def head():
    return BirthHead(('conf',),np.zeros(1),np.ones(1),np.zeros(1),0.)


def detections():
    return IndexedDetections(np.array([[10,10,40,40]],np.float32),np.array([.9]),np.array([2]))


def context(tracker):
    tracker.set_frame_context(gray=np.full((80,80),128,np.uint8),
        prev_propagated=np.empty((0,4),np.float32),frame_size=(80,80))


@pytest.mark.parametrize('factory',[
    lambda:QBirthByteTrack(make_tracker_args(),head=head(),tau=.4),
    lambda:DeferredBirthByteTrack(make_tracker_args(),head=head(),tau=.6,confirm_hits=3),
    lambda:make_deferred_cascade(make_tracker_args(),buffer=.1,cascade_buffer=.2,head=head(),tau=.6,confirm_hits=3),
])
def test_birth_reset_clears_frame_context_and_logs(factory):
    tracker=factory()
    context(tracker)
    tracker.update(detections())
    assert tracker.birth_log
    tracker.reset()
    assert tracker._context is None
    assert tracker.birth_log == tracker.events == tracker.predictions == []
    assert tracker.stage == ''
    with pytest.raises(RuntimeError,match='set_frame_context'):
        tracker.update(detections())
    assert tracker.frame_id == 0
    assert tracker.tracked_stracks == []


def test_missing_birth_context_fails_before_mutating_frame_counter():
    tracker=QBirthByteTrack(make_tracker_args(),head=head(),tau=.4)
    with pytest.raises(RuntimeError,match='set_frame_context'):
        tracker.update(detections())
    assert tracker.frame_id == 0
    context(tracker)
    actual=tracker.update(detections())
    fresh=QBirthByteTrack(make_tracker_args(),head=head(),tau=.4)
    context(fresh)
    np.testing.assert_array_equal(actual,fresh.update(detections()))


def test_motion_reset_does_not_apply_previous_videos_flow():
    tracker=MotionAugmentedByteTrack(make_tracker_args(),blend=1.,use_confidence=True)
    tracker.set_motion_field(MotionField(np.array([[45,45],[50,50],[55,55]],np.float32),
                                        np.tile([[8.,3.]],(3,1))))
    tracker.motion_audit.append({'previous':True})
    tracker.reset()
    track=STrack(np.array([50,50,20,20,0],np.float32),.9,2)
    track.activate(KalmanFilterXYAH(),frame_id=1)
    old=track.mean[:2].copy()
    tracker.multi_predict([track])
    np.testing.assert_array_equal(track.mean[:2],old)
    assert all('previous' not in row for row in tracker.motion_audit)


def test_motion_source_reset_clears_derived_state_and_statistics():
    tracker=MotionSourceByteTrack(make_tracker_args())
    tracker._previous_xywh={7:np.ones(4)};tracker._p_bad={7:.9}
    tracker.stats={'switched':8,'tracks_seen':12}
    tracker.reset()
    assert not tracker._previous_xywh and not tracker._p_bad
    assert tracker.stats == {'switched':0,'tracks_seen':0}


@pytest.mark.parametrize('use_confidence',[False,True])
def test_empty_motion_is_absence_of_measurement_not_stationary_object(use_confidence):
    tracker=MotionAugmentedByteTrack(make_tracker_args(),blend=1.,use_confidence=use_confidence)
    track=STrack(np.array([50,50,20,20,0],np.float32),.9,2)
    track.activate(KalmanFilterXYAH(),frame_id=1)
    track.mean[4:6]=[4.,-2.]
    before=track.mean[:2].copy()
    tracker.multi_predict([track])
    np.testing.assert_array_equal(track.mean[:2],before+[4.,-2.])
    assert tracker.motion_audit[-1]['weight']==0
