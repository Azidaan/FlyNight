"""Rejected caller input must not advance frame counters or external estimators."""
import copy

import numpy as np
import pytest

from tracking.flyvis_tracker import FlyVisTracker, MotionField


class CountingEstimator:
    def reset(self):
        self.calls = 0
    def observe(self, frame):
        self.calls += 1
        return MotionField.empty()


def update(tracker, **changes):
    args = dict(xyxy=np.array([[0, 0, 10, 10]]), confidence=np.array([.8]),
                class_id=np.array([2]), frame=np.zeros((16, 16), np.uint8))
    args.update(changes)
    return tracker.update(**args)


@pytest.mark.parametrize("changes", [
    dict(xyxy=np.array([[0, 0, np.nan, 10]])),
    dict(confidence=np.array([])),
    dict(confidence=np.array([np.nan])),
    dict(class_id=np.array([np.nan])),
    dict(class_id=np.array([np.inf])),
    dict(class_id=np.array([-1])),
    dict(class_id=np.array([2.5])),
    dict(class_id=np.array([2**63], dtype=np.uint64)),
    dict(frame=None),
    dict(motion_field=MotionField.empty()),
])
def test_invalid_request_keeps_tracker_and_estimator_at_previous_frame(changes):
    estimator, control_estimator = CountingEstimator(), CountingEstimator()
    tracker, control = FlyVisTracker(motion_estimator=estimator), FlyVisTracker(motion_estimator=control_estimator)
    update(tracker)
    update(control)
    before_events = copy.deepcopy(tracker.events)
    before_history = copy.deepcopy(tracker.motion_history)
    before_box = tracker.tracks[0].box.copy()
    with pytest.raises(ValueError):
        update(tracker, **changes)
    assert tracker.frame_index == 0 and estimator.calls == 1
    assert tracker.events == before_events and tracker.motion_history == before_history
    np.testing.assert_array_equal(tracker.tracks[0].box, before_box)
    np.testing.assert_array_equal(update(tracker), update(control))
    assert tracker.events == control.events and tracker.motion_history == control.motion_history
