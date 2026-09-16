"""Цель 4: правила меток и признаков выборок событий (карточка event_dataset_v1)."""

from __future__ import annotations

import numpy as np
import pytest

from tracking.event_dataset import (
    IdentityTimeline,
    admissible_candidates,
    alive_label,
    birth_label,
    frames_to_exit,
    image_box_stats,
)


def test_identity_timeline_uses_only_reference_frames_strictly_before() -> None:
    timeline = IdentityTimeline()
    timeline.record(ref_index=2, track_id=7, gt_id="a")
    timeline.record(ref_index=5, track_id=7, gt_id="b")
    assert timeline.before(7, 2) is None
    assert timeline.before(7, 3) == "a"
    assert timeline.before(7, 5) == "a"
    assert timeline.upto(7, 5) == "b"
    assert timeline.before(99, 10) is None


def test_birth_label_requires_match_and_persistence_over_horizon() -> None:
    presence = {"g": {3, 4, 5, 6}, "h": {3}}
    assert birth_label(first_ref=3, gt_id="g", presence=presence, horizon=3, n_ref=10) == 1
    assert birth_label(first_ref=3, gt_id="h", presence=presence, horizon=3, n_ref=10) == 0
    assert birth_label(first_ref=3, gt_id=None, presence=presence, horizon=3, n_ref=10) == 0
    assert birth_label(first_ref=8, gt_id="g", presence=presence, horizon=3, n_ref=10) is None  # цензура: 8+2 = 10 за концом


def test_alive_label_looks_for_object_return_in_window_with_censoring() -> None:
    presence = {"g": {1, 2, 3, 9}}
    assert alive_label(lost_after_ref=3, gt_id="g", presence=presence, window=6, n_ref=20) == 1  # 9 ∈ (3, 9]
    assert alive_label(lost_after_ref=3, gt_id="g", presence=presence, window=5, n_ref=20) == 0
    assert alive_label(lost_after_ref=16, gt_id="g", presence=presence, window=6, n_ref=20) is None  # окно за концом


def test_admissible_candidates_are_class_aware_and_geometry_gated() -> None:
    predictions = np.asarray([[10, 10, 30, 30], [200, 200, 240, 240]], np.float32)
    prediction_classes = np.asarray([2, 2])
    detections = np.asarray([[12, 12, 32, 32], [45, 45, 65, 65], [205, 205, 245, 245], [12, 12, 32, 32]], np.float32)
    detection_classes = np.asarray([2, 2, 2, 0])
    candidates = admissible_candidates(predictions, prediction_classes, detections, detection_classes, distance_scale=2.5)
    assert candidates[0] == [0, 1]  # 1 — без пересечения, центр на 49,5 px < 2,5·20
    assert candidates[1] == [2]


def test_image_box_stats_and_frames_to_exit() -> None:
    gray = np.zeros((100, 100), np.uint8)
    gray[:, 50:] = 255
    stats = image_box_stats(gray, (40, 0, 60, 100))
    assert stats["mean"] == pytest.approx(127.5, abs=1)
    assert stats["dark_fraction"] == pytest.approx(0.5, abs=0.05)
    assert stats["bright_fraction"] == pytest.approx(0.5, abs=0.05)
    assert frames_to_exit((90, 50, 100, 60), velocity=(5.0, 0.0), frame_size=(120, 100)) == pytest.approx(4.0)
    assert frames_to_exit((90, 50, 100, 60), velocity=(0.0, 0.0), frame_size=(120, 100)) == np.inf
