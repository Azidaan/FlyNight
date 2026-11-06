"""Classical flow owns its previous image and commits only successful frames."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tracking.classical_motion import OpenCVMotionEstimator


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_first_frame_buffer_is_owned_by_estimator(method):
    estimator = OpenCVMotionEstimator(method)
    frame = np.full((32, 48), 17, np.uint8)
    estimator.observe(frame)
    frame[:] = 200
    np.testing.assert_array_equal(estimator._previous, np.full(frame.shape, 17, np.uint8))


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_invalid_flow_preserves_previous_frame_and_retry(method, monkeypatch):
    estimator = OpenCVMotionEstimator(method)
    first = np.full((32, 48), 17, np.uint8)
    next_frame = np.full((32, 48), 80, np.uint8)
    attempts = []
    def calculate(previous, current, *args, **kwargs):
        attempts.append((previous.copy(), current.copy()))
        return np.full((*previous.shape, 2), np.nan if len(attempts) == 1 else 1., np.float32)
    if method == "farneback":
        monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", calculate)
    else:
        estimator._dis = SimpleNamespace(calc=calculate)
    estimator.observe(first)
    before_stats = dict(estimator.stats)
    with pytest.raises(RuntimeError, match="NaN|Inf"):
        estimator.observe(next_frame)
    np.testing.assert_array_equal(estimator._previous, first)
    assert estimator.stats == before_stats
    estimator.observe(next_frame)
    np.testing.assert_array_equal(attempts[-1][0], first)
    next_frame[:] = 240
    np.testing.assert_array_equal(estimator._previous, np.full(first.shape, 80, np.uint8))


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_original_shape_change_cannot_hide_in_resize_rounding(method):
    estimator = OpenCVMotionEstimator(method, processing_scale=.25)
    first = np.zeros((100, 100), np.uint8)
    estimator.observe(first)
    with pytest.raises(ValueError, match="reset"):
        estimator.observe(np.zeros((101, 101), np.uint8))
    assert estimator.stats["observations"] == 1
    estimator.reset()
    assert len(estimator.observe(np.zeros((101, 101), np.uint8)).points_xy) > 0


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_reused_grayscale_buffer_matches_independent_native_frames(method):
    first = np.random.default_rng(52).integers(0, 256, (64, 96), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (0, 0), 1.)
    second = cv2.warpAffine(first, np.array([[1, 0, 3], [0, 1, 0]], np.float32),
                            (96, 64), borderMode=cv2.BORDER_REFLECT)
    control, reused = OpenCVMotionEstimator(method), OpenCVMotionEstimator(method)
    control.observe(first)
    expected = control.observe(second)
    buffer = first.copy()
    reused.observe(buffer)
    buffer[:] = second
    actual = reused.observe(buffer)
    np.testing.assert_array_equal(actual.points_xy, expected.points_xy)
    np.testing.assert_allclose(actual.vectors_xy, expected.vectors_xy, rtol=1e-6, atol=1e-6)
