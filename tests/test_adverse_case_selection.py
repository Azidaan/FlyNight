from __future__ import annotations

import numpy as np

from scripts.select_adverse_video_cases import annotation_statistics, sampled


def test_sampled_keeps_endpoints() -> None:
    paths = list(range(20))
    result = sampled(paths, 5)
    assert result[0] == 0
    assert result[-1] == 19
    assert len(result) == 5


def test_annotation_statistics_counts_occlusion_transitions() -> None:
    def label(occluded: bool) -> dict:
        return {"id": "1", "attributes": {"occluded": occluded}}

    frames = [
        {"labels": [label(False)]},
        {"labels": [label(True)]},
        {"labels": [label(False)]},
    ]
    stats = annotation_statistics(frames)
    assert stats["occluded_labels"] == 1
    assert stats["occlusion_transitions"] == 2
