"""Цель 2, фовеация: квадратная вырезка вокруг предсказания и банк глаз FlyVis на треки."""

from __future__ import annotations

import numpy as np
import pytest

from tracking.foveated_eyes import (
    crop_box,
    original_detection_index,
    box_to_crop_frame,
)


def test_crop_box_is_square_with_context_and_minimum_side() -> None:
    x1, y1, x2, y2 = crop_box((100.0, 100.0, 20.0, 10.0), frame_size=(1280, 720), context=2.0, min_side=64)
    assert (x2 - x1) == (y2 - y1) == 64  # 2*max(20,10)=40 < 64
    assert (x1 + x2) / 2 == pytest.approx(100.0) and (y1 + y2) / 2 == pytest.approx(100.0)
    x1, y1, x2, y2 = crop_box((640.0, 360.0, 200.0, 80.0), frame_size=(1280, 720), context=2.0, min_side=64)
    assert (x2 - x1) == (y2 - y1) == 400


def test_crop_box_is_shifted_inside_frame_without_resizing() -> None:
    x1, y1, x2, y2 = crop_box((10.0, 10.0, 50.0, 50.0), frame_size=(1280, 720), context=2.0, min_side=64)
    assert x1 == 0 and y1 == 0 and x2 == 100 and y2 == 100


def test_crop_box_larger_than_frame_keeps_square_of_frame_height() -> None:
    x1, y1, x2, y2 = crop_box((640.0, 360.0, 1000.0, 600.0), frame_size=(1280, 720), context=2.0, min_side=64)
    assert (x1, y1, x2, y2) == (280, 0, 1000, 720)  # квадрат 720, без искажения аспекта


def test_box_to_crop_frame_scales_into_eye_pixels() -> None:
    local = box_to_crop_frame((110, 120, 130, 140), crop=(100, 100, 200, 200), eye_size=391)
    np.testing.assert_allclose(local, (39.1, 78.2, 117.3, 156.4))


def test_original_detection_index_maps_subset_index_back() -> None:
    conf = np.asarray([0.9, 0.15, 0.3, 0.2, 0.5], np.float32)
    # high subset (>=0.25): [0, 2, 4]; low subset (0.1 < c < 0.25): [1, 3]
    assert original_detection_index(conf, subset_index=1, score=0.3, high=0.25, low=0.1) == 2
    assert original_detection_index(conf, subset_index=1, score=0.2, high=0.25, low=0.1) == 3
    assert original_detection_index(conf, subset_index=2, score=0.5, high=0.25, low=0.1) == 4


def test_eye_bank_births_steps_and_removes_eyes_deterministically() -> None:
    from tracking.flyvis_motion import FlyVisMotionEstimator, TileLayout
    from tracking.foveated_eyes import EyeBank

    estimator = FlyVisMotionEstimator(mode="tiled", layout=TileLayout(2, 3, 0.20), source_fps=30.0, dt=0.02, flow_gain=1.0, temporal_resampling="linear")
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (180, 320), dtype=np.uint8) for _ in range(3)]

    def run(order):
        bank = EyeBank(estimator, context=2.0, min_side=64, fade_in_seconds=0.2)
        bank.birth(frames[0], {tid: (60.0, 60.0, 30.0, 20.0) for tid in order})
        first = bank.step(frames[1], {tid: (62.0, 60.0, 30.0, 20.0) for tid in order})
        bank.birth(frames[1], {3: (200.0, 100.0, 40.0, 40.0)})
        second = bank.step(frames[2], {tid: (64.0, 60.0, 30.0, 20.0) for tid in order} | {3: (201.0, 100.0, 40.0, 40.0)})
        return bank, first, second

    bank, first, second = run([1, 2])
    assert set(first) == {1, 2} and set(second) == {1, 2, 3}
    assert first[1].shape == (34, 721) and bank.baseline(1).shape == (34, 721)
    np.testing.assert_allclose(first[1], first[2], atol=1e-5)  # одинаковые глаза -> одинаковая активность
    bank.remove({2})
    assert set(bank.track_ids) == {1, 3}
    _, first_b, second_b = run([2, 1])
    np.testing.assert_allclose(second[1], second_b[1], atol=1e-4)  # порядок в батче не важен
    assert np.isfinite(second[3]).all()
