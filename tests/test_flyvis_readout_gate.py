"""Цель 2: гейт считывания внутренней активности FlyVis для q_match."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_flyvis_readout_gate import (
    READOUT_PAIR_FEATURES,
    frame_activity_matrix,
    readout_pair_features,
    shuffle_within_video,
)
from tracking.flyvis_activity import OUTPUT_CELL_TYPES


def test_frame_activity_matrix_is_tile_major_like_points() -> None:
    tiles, types, columns = 2, 3, 4
    frame = np.arange(tiles * types * columns, dtype=np.float16).reshape(tiles, types, columns)
    matrix = frame_activity_matrix(frame)
    assert matrix.shape == (types, tiles * columns)
    # тип 0: сначала колонки фрагмента 0, затем фрагмента 1
    np.testing.assert_array_equal(matrix[0, :columns], frame[0, 0])
    np.testing.assert_array_equal(matrix[0, columns:], frame[1, 0])


def _points() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(0, 100, 10, dtype=np.float32), np.arange(0, 100, 10, dtype=np.float32))
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_readout_pair_features_has_declared_names_and_pair_terms() -> None:
    points = _points()
    types = len(OUTPUT_CELL_TYPES)
    rng = np.random.default_rng(0)
    previous = rng.random((types, len(points))).astype(np.float32)
    current = rng.random((types, len(points))).astype(np.float32)
    baseline = np.zeros_like(previous)
    values = readout_pair_features(
        previous, current, baseline, points,
        previous_box=(20, 20, 40, 40), current_box=(22, 21, 42, 41), min_margin=10.0,
    )
    assert values.shape == (len(READOUT_PAIR_FEATURES),)
    assert np.isfinite(values).all()
    names = list(READOUT_PAIR_FEATURES)
    for name in ("prev_inside_entropy", "cur_inside_entropy", "pair_net_cosine", "pair_entropy_delta", "pair_energy_delta"):
        assert name in names


def test_readout_pair_features_cosine_is_one_for_identical_activity() -> None:
    points = _points()
    types = list(OUTPUT_CELL_TYPES)
    activity = np.zeros((len(types), len(points)), np.float32)
    activity[types.index("T4a")] = 1.0
    values = readout_pair_features(
        activity, activity, np.zeros_like(activity), points,
        previous_box=(20, 20, 40, 40), current_box=(20, 20, 40, 40), min_margin=10.0,
    )
    index = list(READOUT_PAIR_FEATURES).index("pair_net_cosine")
    assert values[index] == pytest.approx(1.0, abs=1e-6)


def test_shuffle_within_video_permutes_rows_only_inside_each_video() -> None:
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    video = np.asarray([0, 0, 0, 1, 1, 1])
    shuffled = shuffle_within_video(matrix, video, seed=0)
    assert shuffled.shape == matrix.shape
    for v in (0, 1):
        np.testing.assert_array_equal(np.sort(shuffled[video == v], axis=0), np.sort(matrix[video == v], axis=0))
    assert not np.array_equal(shuffled, matrix)
    np.testing.assert_array_equal(shuffle_within_video(matrix, video, seed=0), shuffled)
