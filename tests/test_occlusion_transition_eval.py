from __future__ import annotations

from evaluation.occlusion_transition_eval import evaluate_occlusion_transitions


def label(object_id: str, prediction_id: str | None = None, *, occluded: bool = False) -> dict:
    return {
        "id": prediction_id if prediction_id is not None else object_id,
        "category": "car",
        "attributes": {"occluded": occluded},
        "box2d": {"x1": 0, "y1": 0, "x2": 20, "y2": 20},
    }


def frame(labels: list[dict]) -> dict:
    return {"labels": labels}


def test_counts_retained_and_switched_occlusion_transitions() -> None:
    truth = [
        frame([label("gt", occluded=False)]),
        frame([label("gt", occluded=True)]),
        frame([label("gt", occluded=False)]),
    ]
    result = [
        frame([label("unused", "track-1")]),
        frame([label("unused", "track-1")]),
        frame([label("unused", "track-2")]),
    ]
    metrics = evaluate_occlusion_transitions(truth, result)
    occlusion = metrics["touches_occlusion"]
    assert occlusion["eligible"] == 2
    assert occlusion["matched_both"] == 2
    assert occlusion["retained"] == 1
    assert occlusion["switched"] == 1
    assert occlusion["retention_given_matched"] == 0.5


def test_unmatched_result_reduces_end_to_end_retention() -> None:
    truth = [frame([label("gt")]), frame([label("gt")])]
    result = [frame([label("unused", "track-1")]), frame([])]
    metrics = evaluate_occlusion_transitions(truth, result)["visible_visible"]
    assert metrics["eligible"] == 1
    assert metrics["matched_both"] == 0
    assert metrics["retained_over_eligible"] == 0.0
