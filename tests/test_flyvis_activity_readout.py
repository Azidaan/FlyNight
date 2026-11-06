"""Цель 2: кэш внутренней активности FlyVis и объектно-центрированное считывание."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tracking.flyvis_activity import (
    DIRECTION_SUFFIXES,
    OUTPUT_CELL_TYPES,
    box_readout_t4t5,
    load_activity_cache,
    mean_activity_by_type,
    save_activity_cache,
    column_spacing,
    select_columns,
    type_column_indices,
)


def _toy_types() -> np.ndarray:
    # три колонки, порядок узлов перемешан между типами, как в реальном коннектоме
    return np.asarray(
        ["T4a", "T4a", "T4a", "R1", "R1", "R1", "T5a", "T5a", "T5a", "Am"], dtype=str
    )


def test_type_column_indices_returns_one_index_array_per_requested_type() -> None:
    indices = type_column_indices(_toy_types(), ["T4a", "T5a"])
    assert list(indices) == ["T4a", "T5a"]
    np.testing.assert_array_equal(indices["T4a"], [0, 1, 2])
    np.testing.assert_array_equal(indices["T5a"], [6, 7, 8])


def test_type_column_indices_rejects_types_with_different_column_counts() -> None:
    with pytest.raises(ValueError):
        type_column_indices(_toy_types(), ["T4a", "Am"])


def test_mean_activity_by_type_averages_internal_steps_and_orders_types() -> None:
    tiles, steps, nodes = 2, 2, 10
    activity = np.zeros((tiles, steps, nodes), np.float32)
    activity[0, 0, 0:3] = [1, 2, 3]
    activity[0, 1, 0:3] = [3, 4, 5]
    activity[1, :, 6:9] = 7
    indices = type_column_indices(_toy_types(), ["T4a", "T5a"])
    mean = mean_activity_by_type(activity, indices)
    assert mean.shape == (tiles, 2, 3)
    assert mean.dtype == np.float16
    np.testing.assert_allclose(mean[0, 0], [2, 3, 4])
    np.testing.assert_allclose(mean[1, 1], [7, 7, 7])


def test_activity_cache_roundtrip_preserves_shapes_and_types(tmp_path: Path) -> None:
    frames = np.random.default_rng(0).random((5, 2, 3, 4)).astype(np.float16)
    baseline = frames[0]
    points = np.arange(8, dtype=np.float32).reshape(4, 2)
    types = ["T4a", "T4b", "T5a"]
    save_activity_cache(tmp_path / "act.npz", frames=frames, baseline=baseline, points=points, cell_types=types)
    cache = load_activity_cache(tmp_path / "act.npz")
    assert cache["frames"].shape == (5, 2, 3, 4)
    assert cache["baseline"].shape == (2, 3, 4)
    assert list(cache["cell_types"]) == types
    np.testing.assert_array_equal(cache["points"], points)


def test_output_cell_types_contain_eight_direction_channels() -> None:
    assert [f"T4{s}" for s in DIRECTION_SUFFIXES] == ["T4a", "T4b", "T4c", "T4d"]
    for suffix in DIRECTION_SUFFIXES:
        assert f"T4{suffix}" in OUTPUT_CELL_TYPES
        assert f"T5{suffix}" in OUTPUT_CELL_TYPES
    assert len(OUTPUT_CELL_TYPES) == 34


def _grid_points() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(0, 100, 10, dtype=np.float32), np.arange(0, 100, 10, dtype=np.float32))
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_select_columns_splits_inside_and_ring() -> None:
    points = _grid_points()
    assert column_spacing(points) == pytest.approx(10.0)
    inside, ring = select_columns(points, box=(20, 20, 40, 40), ring_scale=1.5, min_margin=column_spacing(points))
    assert inside.size == 9  # x,y ∈ {20,30,40}
    assert ring.size == 16  # кольцо 10..50 минус внутренние 3x3
    assert not np.intersect1d(inside, ring).size


def test_select_columns_falls_back_to_nearest_column_for_tiny_box() -> None:
    inside, _ = select_columns(_grid_points(), box=(33, 33, 34, 34), ring_scale=1.5)
    assert inside.size == 1
    assert inside[0] == np.argmin(np.abs(_grid_points() - [33.5, 33.5]).sum(1))


def _activity_with_direction(points: np.ndarray, dominant: str, inside_box: tuple) -> np.ndarray:
    """Активность (types, columns): внутри рамки доминирует направление dominant."""
    types = list(OUTPUT_CELL_TYPES)
    activity = np.zeros((len(types), len(points)), np.float32)
    x1, y1, x2, y2 = inside_box
    inside = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    for suffix in DIRECTION_SUFFIXES:
        level = 1.0 if suffix == dominant else 0.1
        activity[types.index(f"T4{suffix}"), inside] = level
        activity[types.index(f"T5{suffix}"), inside] = level
    return activity


def test_box_readout_t4t5_reports_dominant_direction_and_low_entropy() -> None:
    points = _grid_points()
    activity = _activity_with_direction(points, "a", (20, 20, 40, 40))
    baseline = np.zeros_like(activity)
    features = box_readout_t4t5(activity, baseline, points, box=(20, 20, 40, 40))
    assert features["inside_energy_a"] > features["inside_energy_b"]
    assert 0.0 <= features["inside_entropy"] < 0.7
    assert features["inside_net_x"] > 0
    assert features["coverage"] > 0
    assert features["contrast"] > 0


def test_box_readout_t4t5_uniform_directions_give_maximal_entropy() -> None:
    points = _grid_points()
    types = list(OUTPUT_CELL_TYPES)
    activity = np.zeros((len(types), len(points)), np.float32)
    for suffix in DIRECTION_SUFFIXES:
        activity[types.index(f"T4{suffix}")] = 1.0
    features = box_readout_t4t5(activity, np.zeros_like(activity), points, box=(20, 20, 40, 40))
    assert features["inside_entropy"] == pytest.approx(1.0, abs=1e-6)
    assert features["inside_net_x"] == pytest.approx(0.0, abs=1e-6)


def test_box_readout_t4t5_subtracts_baseline() -> None:
    points = _grid_points()
    activity = _activity_with_direction(points, "a", (20, 20, 40, 40))
    features_zero = box_readout_t4t5(activity, activity, points, box=(20, 20, 40, 40))
    assert features_zero["inside_energy_total"] == pytest.approx(0.0, abs=1e-6)


def test_estimator_exposes_internal_activity_of_current_frame_only() -> None:
    """Настоящая сеть FlyVis: после кадра доступны baseline и активность внутренних шагов."""
    from tracking.flyvis_adapter import DEFAULT_DATA_DIR, DEFAULT_RUNTIME_DIR
    from tracking.flyvis_motion import FlyVisMotionEstimator, TileLayout

    if not (DEFAULT_RUNTIME_DIR / "flyvis").is_dir():
        pytest.skip(f"Локальный FlyVis runtime отсутствует: {DEFAULT_RUNTIME_DIR / 'flyvis'}")
    checkpoint_dir = DEFAULT_DATA_DIR / "results" / "flow" / "0000" / "000" / "chkpts"
    if not any(path.is_file() for path in checkpoint_dir.glob("chkpt_*")):
        pytest.skip(f"Нет локальных контрольных точек chkpt_*: {checkpoint_dir}")

    estimator = FlyVisMotionEstimator(
        mode="tiled", layout=TileLayout(2, 3, 0.20), source_fps=30.0, dt=0.02,
        flow_gain=1.0, temporal_resampling="linear", flow_aggregation="mean",
        device="cpu",
    )
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (180, 320), dtype=np.uint8) for _ in range(2)]
    estimator.observe(frames[0])
    baseline = estimator.baseline_activity
    assert baseline.shape == (6, len(estimator.network.connectome.nodes.type))
    assert estimator.last_activity is None
    estimator.observe(frames[1])
    activity = estimator.last_activity
    assert activity.shape == (6, estimator.integration_steps, baseline.shape[1])
    assert np.isfinite(activity).all()
