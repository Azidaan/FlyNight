from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_flyvis_qmatch_gate import (
    FEATURE_NAMES,
    MatchedDetection,
    box_pair_geometry,
    decision_stratum_metrics,
    detections_with_truth_status,
    interpolate_truth_frame,
    match_detections_to_truth,
    pair_features,
    pair_target,
    paired_video_bootstrap,
    propagate_box,
    temporal_transitions,
)


def truth_frame() -> dict:
    return {
        "labels": [
            {
                "id": "car-a",
                "category": "car",
                "attributes": {"occluded": True, "crowd": False},
                "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
            },
            {
                "id": "crowd",
                "category": "car",
                "attributes": {"occluded": False, "crowd": True},
                "box2d": {"x1": 20, "y1": 0, "x2": 30, "y2": 10},
            },
        ]
    }


def test_matching_is_class_aware_and_excludes_crowd() -> None:
    matched = match_detections_to_truth(
        np.asarray([[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10]]),
        np.asarray([0.8, 0.9, 0.95]),
        np.asarray([2, 2, 0]),
        truth_frame(),
        min_iou=0.5,
    )
    assert len(matched) == 1
    assert matched[0].gt_id == "car-a"
    assert matched[0].detection_index == 0
    assert matched[0].occluded is True


def test_all_current_candidates_keep_unmatched_detections() -> None:
    detections = detections_with_truth_status(
        np.asarray([[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 10]]),
        np.asarray([0.8, 0.9, 0.95]),
        np.asarray([2, 2, 0]),
        truth_frame(),
        min_iou=0.5,
    )
    assert len(detections) == 3
    assert [detection.gt_id for detection in detections] == ["car-a", None, None]


def test_unmatched_detection_cannot_form_positive_pair() -> None:
    previous = MatchedDetection(
        0, "car-a", 2, np.asarray([0, 0, 10, 10]), 0.4, False
    )
    unmatched = MatchedDetection(
        1, None, 2, np.asarray([1, 0, 11, 10]), 0.3, False
    )
    assert pair_target(previous, unmatched) == 0
    assert pair_target(unmatched, unmatched) == 0


def test_decision_stratum_reports_acceptance_without_both_classes() -> None:
    result = decision_stratum_metrics(
        np.asarray([0, 0, 0]), np.asarray([0.1, 0.4, 0.8]), threshold=0.5
    )
    assert result["samples"] == 3
    assert result["positives"] == 0
    assert result["accepted"] == 1
    assert result["acceptance_rate"] == pytest.approx(1 / 3)
    assert result["brier"] == pytest.approx((0.01 + 0.16 + 0.64) / 3)


def test_interpolation_uses_only_common_clean_tracks() -> None:
    left = truth_frame()
    right = {
        "labels": [
            {
                "id": "car-a",
                "category": "car",
                "attributes": {
                    "occluded": False,
                    "truncated": False,
                    "crowd": False,
                },
                "box2d": {"x1": 6, "y1": 0, "x2": 16, "y2": 10},
            },
            {
                "id": "new-car",
                "category": "car",
                "attributes": {
                    "occluded": False,
                    "truncated": False,
                    "crowd": False,
                },
                "box2d": {"x1": 40, "y1": 0, "x2": 50, "y2": 10},
            },
        ]
    }
    left["labels"][0]["attributes"]["truncated"] = False
    left["labels"][0]["attributes"]["occluded"] = False
    interpolated = interpolate_truth_frame(left, right, 0.5)
    assert [label["id"] for label in interpolated["labels"]] == ["car-a"]
    assert interpolated["labels"][0]["box2d"] == {
        "x1": 3.0,
        "y1": 0.0,
        "x2": 13.0,
        "y2": 10.0,
    }
    assert interpolated["labels"][0]["attributes"]["pseudo_interpolated"] is True


def test_adjacent_temporal_mode_expands_reference_interval() -> None:
    left = truth_frame()
    right = truth_frame()
    alignment = [{"video_frame_index": 10}, {"video_frame_index": 13}]
    transitions = list(temporal_transitions([left, right], alignment, "adjacent_pseudo"))
    assert [(row[0], row[1]) for row in transitions] == [(10, 11), (11, 12), (12, 13)]


def test_constant_motion_propagation_accumulates_raw_frames() -> None:
    points = np.asarray([[2, 2], [5, 5], [8, 8]], dtype=np.float32)
    vectors = np.zeros((4, 3, 2), dtype=np.float32)
    vectors[1:, :, 0] = 2.0
    predicted, weight = propagate_box(
        np.asarray([0, 0, 10, 10]), points, vectors, 0, 3
    )
    np.testing.assert_allclose(predicted, [6, 0, 16, 10])
    assert weight == 1.0


def test_geometry_is_normalized_by_previous_box_scale() -> None:
    overlap, distance, size_change = box_pair_geometry(
        np.asarray([0, 0, 10, 10]), np.asarray([10, 0, 20, 10])
    )
    assert overlap == 0.0
    assert distance == 1.0
    assert size_change == 0.0


def test_model_features_do_not_contain_ground_truth_fields() -> None:
    forbidden = ("gt", "identity", "occluded", "target")
    assert not any(token in name for name in FEATURE_NAMES for token in forbidden)


def test_pair_features_mark_class_without_using_identity() -> None:
    previous = MatchedDetection(0, "secret-a", 2, np.asarray([0, 0, 10, 10]), 0.4, False)
    current = MatchedDetection(1, "secret-b", 2, np.asarray([1, 0, 11, 10]), 0.3, True)
    predictions = {
        (0, source): (np.asarray([1, 0, 11, 10]), 0.5)
        for source in ("dis", "flyvis_full", "flyvis_tiled")
    }
    features = pair_features(previous, current, predictions)
    assert features.shape == (len(FEATURE_NAMES),)
    assert features[FEATURE_NAMES.index("class_2")] == 1.0
    assert features[FEATURE_NAMES.index("flyvis_tiled_predicted_iou")] == 1.0


def test_paired_bootstrap_uses_video_as_unit() -> None:
    reference = {
        "test_by_video": {
            "a": {"average_precision": 0.7},
            "b": {"average_precision": 0.8},
            "c": {"average_precision": 0.9},
        }
    }
    candidate = {
        "test_by_video": {
            "a": {"average_precision": 0.71},
            "b": {"average_precision": 0.82},
            "c": {"average_precision": 0.93},
        }
    }
    result = paired_video_bootstrap(
        reference, candidate, "average_precision", repetitions=1000
    )
    assert result["video_count"] == 3
    assert result["mean_difference"] == pytest.approx(0.02)
    assert result["ci95_percentile"][0] > 0.0
