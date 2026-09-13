from __future__ import annotations

import numpy as np
import pytest
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

from tracking.flyvis_tracker import MotionField
from tracking.motion_augmented_bytetrack import MotionAugmentedByteTrack
from tracking.offline_association import make_tracker_args


def uniform_field(dx: float, dy: float) -> MotionField:
    points = np.asarray([[45, 45], [50, 50], [55, 55]], np.float32)
    vectors = np.tile(np.asarray([[dx, dy]], np.float32), (3, 1))
    return MotionField(points, vectors)


def active_track() -> STrack:
    track = STrack(np.asarray([50, 50, 20, 20, 0], np.float32), 0.9, 2)
    track.activate(KalmanFilterXYAH(), frame_id=1)
    return track


def test_external_motion_replaces_zero_velocity_prediction() -> None:
    tracker = MotionAugmentedByteTrack(
        make_tracker_args(), blend=1.0, use_confidence=True
    )
    track = active_track()
    before = track.mean[:2].copy()
    tracker.set_motion_field(uniform_field(3.0, -2.0))
    tracker.multi_predict([track])
    np.testing.assert_allclose(track.mean[:2], before + [3.0, -2.0], atol=1e-5)


def test_lost_only_does_not_change_tracked_prediction() -> None:
    tracker = MotionAugmentedByteTrack(
        make_tracker_args(), blend=1.0, apply_to="lost", use_confidence=False
    )
    track = active_track()
    before = track.mean[:2].copy()
    tracker.set_motion_field(uniform_field(7.0, 4.0))
    tracker.multi_predict([track])
    np.testing.assert_allclose(track.mean[:2], before, atol=1e-5)


@pytest.mark.parametrize("blend", [-0.01, 1.01])
def test_invalid_blend_rejected(blend: float) -> None:
    with pytest.raises(ValueError):
        MotionAugmentedByteTrack(make_tracker_args(), blend=blend)
