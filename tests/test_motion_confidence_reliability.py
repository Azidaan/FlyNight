from __future__ import annotations

import numpy as np
import pytest

from scripts.run_bdd_video_rate_experiment import motion_confidence_reliability


def test_high_confidence_selection_has_lower_error() -> None:
    audit = motion_confidence_reliability(
        confidence=[0.9, 0.8, 0.2, 0.1],
        errors=[1.0, 2.0, 10.0, 20.0],
        normalized_errors=[0.05, 0.10, 0.50, 1.0],
    )

    assert audit["segments"] == 4
    assert audit["confidence_error_pearson"] < 0
    assert audit["top_confidence_fractions"]["0.50"][
        "mean_endpoint_error_pixels"
    ] == pytest.approx(1.5)
    assert audit["top_confidence_fractions"]["1.00"][
        "mean_endpoint_error_pixels"
    ] == pytest.approx(8.25)


def test_fixed_bins_report_empirical_acceptable_rate() -> None:
    audit = motion_confidence_reliability(
        confidence=[0.15, 0.19, 0.85],
        errors=[3.0, 9.0, 1.0],
        normalized_errors=[0.1, 0.4, 0.05],
    )

    low = audit["bins"][0]
    high = audit["bins"][1]
    assert (low["lower"], low["upper"], low["count"]) == (0.1, 0.2, 2)
    assert low["acceptable_rate_norm_0.25"] == pytest.approx(0.5)
    assert (high["lower"], high["upper"], high["count"]) == (0.8, 0.9, 1)
    assert high["acceptable_rate_norm_0.25"] == pytest.approx(1.0)


def test_empty_reliability_is_explicit() -> None:
    audit = motion_confidence_reliability([], [], [])
    assert audit["segments"] == 0
    assert audit["confidence_error_pearson"] is None
    assert audit["bins"] == []


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        motion_confidence_reliability([0.5], [1.0, 2.0], [0.1])


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        motion_confidence_reliability([np.nan], [1.0], [0.1])
