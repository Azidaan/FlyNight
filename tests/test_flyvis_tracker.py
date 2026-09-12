from __future__ import annotations

import numpy as np
import cv2
import pytest

from tracking.classical_motion import OpenCVMotionEstimator
from tracking.flyvis_motion import TileLayout, fixed_tile_bounds, integration_plan
from tracking.flyvis_tracker import (
    FlyVisTracker,
    FlyVisTrackerConfig,
    MotionField,
    TrackStatus,
)


def motion_field(boxes_and_vectors) -> MotionField:
    points = []
    vectors = []
    for box, vector in boxes_and_vectors:
        x1, y1, x2, y2 = box
        for fx, fy in ((0.3, 0.3), (0.5, 0.5), (0.7, 0.7)):
            points.append([x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)])
            vectors.append(vector)
    return MotionField(np.asarray(points), np.asarray(vectors))


def update(tracker, boxes, *, classes=None, field=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(
        classes if classes is not None else np.full(len(boxes), 2), dtype=np.int64
    )
    scores = np.full(len(boxes), 0.9, dtype=np.float32)
    return tracker.update(
        boxes,
        scores,
        classes,
        motion_field=field if field is not None else MotionField.empty(),
    )


def test_fixed_tiles_cover_frame_and_overlap() -> None:
    bounds = fixed_tile_bounds(1280, 720, TileLayout(rows=2, columns=3, overlap=0.2))
    assert len(bounds) == 6
    assert min(x1 for x1, _, _, _ in bounds) == 0
    assert min(y1 for _, y1, _, _ in bounds) == 0
    assert max(x2 for _, _, x2, _ in bounds) == 1280
    assert max(y2 for _, _, _, y2 in bounds) == 720
    first, second = bounds[0], bounds[1]
    assert first[2] > second[0]


@pytest.mark.parametrize(
    ("source_fps", "expected_steps"),
    ((5.0, 10), (25.0, 2), (29.770992366412212, 2), (50.0, 1), (60.0, 1)),
)
def test_integration_plan_covers_frame_interval_without_large_step(
    source_fps: float, expected_steps: int
) -> None:
    steps, dt = integration_plan(source_fps, 0.02)
    assert steps == expected_steps
    assert dt <= 0.02 + 1e-12
    assert steps * dt == pytest.approx(1.0 / source_fps)


def test_motion_field_keeps_full_and_relative_motion_separate() -> None:
    points = np.array(
        [[2, 2], [5, 5], [8, 8], [12, 5], [15, 5], [-2, 5]], dtype=np.float32
    )
    vectors = np.array(
        [[7, 1], [7, 1], [7, 1], [2, 0], [2, 0], [2, 0]], dtype=np.float32
    )
    estimate = MotionField(points, vectors).estimate_box(
        np.array([0, 0, 10, 10]), context_scale=2.0, min_samples=3
    )
    np.testing.assert_allclose(estimate.vector_xy, [7, 1])
    np.testing.assert_allclose(estimate.background_xy, [2, 0])
    np.testing.assert_allclose(estimate.relative_xy, [5, 1])


def test_flyvis_motion_can_match_without_iou() -> None:
    tracker = FlyVisTracker()
    first_box = [0, 0, 10, 10]
    out0 = update(tracker, [first_box])
    assert out0[:, 4].tolist() == [1]

    field = motion_field([(first_box, [20, 0])])
    out1 = update(tracker, [[20, 0, 30, 10]], field=field)
    assert out1[:, 4].tolist() == [1]
    np.testing.assert_allclose(out1[0, :4], [20, 0, 30, 10])


def test_spatially_inconsistent_flow_is_confidence_weighted() -> None:
    tracker = FlyVisTracker()
    update(tracker, [[0, 0, 10, 10]])
    field = MotionField(
        np.asarray([[2, 2], [5, 5], [8, 8]], dtype=np.float32),
        np.asarray([[100, 0], [50, 0], [0, 0]], dtype=np.float32),
    )
    update(tracker, [[1, 0, 11, 10]], field=field)
    assert tracker.motion_history[-1]["confidence"] < 0.03
    assert abs(tracker.motion_history[-1]["dx"]) < 2.0


def test_crossing_tracks_keep_identity_from_opposite_motion() -> None:
    tracker = FlyVisTracker()
    box_a = [0, 0, 10, 10]
    box_b = [40, 0, 50, 10]
    out0 = update(tracker, [box_a, box_b])
    assert out0[:, 4].tolist() == [1, 2]

    field1 = motion_field([(box_a, [15, 0]), (box_b, [-15, 0])])
    out1 = update(tracker, [[15, 0, 25, 10], [25, 0, 35, 10]], field=field1)
    assert out1[:, 4].tolist() == [1, 2]

    field2 = motion_field(
        [([15, 0, 25, 10], [15, 0]), ([25, 0, 35, 10], [-15, 0])]
    )
    # Обнаружения намеренно перечислены в обратном пространственном порядке.
    out2 = update(tracker, [[10, 0, 20, 10], [30, 0, 40, 10]], field=field2)
    by_id = {int(row[4]): row[:4] for row in out2}
    np.testing.assert_allclose(by_id[1], [30, 0, 40, 10])
    np.testing.assert_allclose(by_id[2], [10, 0, 20, 10])


def test_lost_track_is_refound_with_same_id() -> None:
    tracker = FlyVisTracker(FlyVisTrackerConfig(max_age=2))
    first = [0, 0, 10, 10]
    update(tracker, [first])
    field1 = motion_field([(first, [5, 0])])
    assert len(update(tracker, [], field=field1)) == 0
    assert tracker.active_tracks[0].status == TrackStatus.LOST

    predicted = [5, 0, 15, 10]
    field2 = motion_field([(predicted, [5, 0])])
    out = update(tracker, [[10, 0, 20, 10]], field=field2)
    assert out[:, 4].tolist() == [1]
    assert tracker.active_tracks[0].status == TrackStatus.CONFIRMED
    assert tracker.events[-1]["event"] == "refound"


def test_removed_id_is_never_reused() -> None:
    tracker = FlyVisTracker(FlyVisTrackerConfig(max_age=1))
    update(tracker, [[0, 0, 10, 10]])
    update(tracker, [])
    update(tracker, [])
    assert tracker.tracks[0].status == TrackStatus.REMOVED
    out = update(tracker, [[0, 0, 10, 10]])
    assert out[:, 4].tolist() == [2]


def test_class_mismatch_creates_new_identity() -> None:
    tracker = FlyVisTracker()
    update(tracker, [[0, 0, 10, 10]], classes=[2])
    out = update(tracker, [[0, 0, 10, 10]], classes=[0])
    assert out[:, 4].tolist() == [2]
    assert tracker.tracks[0].status == TrackStatus.LOST


def test_tentative_track_requires_configured_hits() -> None:
    tracker = FlyVisTracker(FlyVisTrackerConfig(min_hits=2))
    assert len(update(tracker, [[0, 0, 10, 10]])) == 0
    out = update(tracker, [[0, 0, 10, 10]])
    assert out[:, 4].tolist() == [1]


def test_constant_velocity_control_is_learned_from_matches() -> None:
    tracker = FlyVisTracker(
        FlyVisTrackerConfig(update_velocity_from_matches=True)
    )
    update(tracker, [[0, 0, 10, 10]])
    update(tracker, [[5, 0, 15, 10]])
    out = update(tracker, [[10, 0, 20, 10]])
    assert out[:, 4].tolist() == [1]
    np.testing.assert_allclose(tracker.active_tracks[0].velocity_xy, [5, 0])


@pytest.mark.parametrize("method", ["farneback", "dis"])
def test_classical_flow_detects_known_horizontal_shift(method: str) -> None:
    rng = np.random.default_rng(42)
    first = rng.integers(0, 256, size=(96, 128), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (0, 0), 1.0)
    matrix = np.array([[1, 0, 5], [0, 1, 0]], dtype=np.float32)
    second = cv2.warpAffine(first, matrix, (128, 96), borderMode=cv2.BORDER_REFLECT)
    estimator = OpenCVMotionEstimator(method, sample_step=4)
    estimator.observe(first)
    field = estimator.observe(second)
    estimate = field.estimate_box(
        np.array([20, 20, 108, 76]), context_scale=1.5, min_samples=3
    )
    assert estimate.vector_xy[0] > 2.0
    assert abs(estimate.vector_xy[1]) < 1.0
