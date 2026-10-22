import numpy as np
import pytest
from tracking.z13_localization_labels import localization_targets


def test_proximity_independent_of_kalman_and_padding():
    rows = [dict(kind=1, gt=[0, 0], pred=[0, 0], scale=8)]
    centers = np.array([[[2., 0.], [2.01, 0.], [0., 0.]]])
    mask = np.array([[True, True, False]])
    assert localization_targets(rows, centers, mask).tolist() == [[1., 0., 0.]]
    rows[0]['pred'] = [1000, 1000]
    assert localization_targets(rows, centers, mask).tolist() == [[1., 0., 0.]]


def test_absent_unknown_and_bad_shape():
    rows = [dict(kind=0), dict(kind=-1)]
    assert not localization_targets(rows, np.zeros((2, 3, 2)), np.ones((2, 3), bool)).any()
    with pytest.raises(ValueError):
        localization_targets(rows, np.zeros((1, 3, 2)), np.ones((2, 3), bool))
