from types import SimpleNamespace
import numpy as np
import pytest
from tracking.z13_proposal_geometry import geometry_actions


def test_center_cannot_fix_wrong_scale_but_full_box_can_and_state_is_restored():
    track = SimpleNamespace(mean=np.array([0., 0., 1., 1., .4, .5]), cls=2)
    detections = [SimpleNamespace(xyxy=np.array([-5., -5., 5., 5.]), cls=2)]
    def cost(tracks, unused):
        a = tracks[0].mean
        intersection = min(a[2], 10) * min(a[3], 10)
        return np.array([[1 - .9 * intersection / (a[2] * a[3] + 100 - intersection)]])
    before = track.mean.copy()
    original = cost([track], detections)[0]
    rows, meta = geometry_actions(track, detections, original, cost, [-20, -20, 20, 20])
    assert rows['centers_eye'][1][0] > .8
    assert rows['boxes_eye'][1][0] < .8
    assert np.array_equal(track.mean, before) and meta['proposals'][0]['inside_eye']
    assert all(np.array_equal(r[0], original) for r in rows.values())


def test_outside_eye_and_other_classes_do_not_enter_observable_eye_actions():
    track = SimpleNamespace(mean=np.arange(8, dtype=float), cls=2)
    detections = [SimpleNamespace(xyxy=np.array([50., 50., 60., 60.]), cls=2),
                  SimpleNamespace(xyxy=np.array([0., 0., 1., 1.]), cls=1)]
    rows, _ = geometry_actions(track, detections, [.2, .3], lambda *_: [[.4, .5]], [0, 0, 10, 10])
    assert len(rows['centers_eye']) == 1 and len(rows['boxes_frame']) == 2
    assert len(rows['oracle_gt_center_eye']) == 1


def test_shadow_error_restores_all_state_components():
    track = SimpleNamespace(mean=np.arange(8, dtype=float), cls=2)
    before = track.mean.copy()
    detection = SimpleNamespace(xyxy=np.array([1., 1., 3., 3.]), cls=2)
    def fail(*_):
        raise RuntimeError('deliberate failure')
    with pytest.raises(RuntimeError):
        geometry_actions(track, [detection], [.2], fail, [0, 0, 10, 10])
    assert np.array_equal(track.mean, before)
