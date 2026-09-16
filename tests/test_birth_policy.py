"""Цель 6: вето рождений по q_birth не трогает ассоциацию и воспроизводит стенд при τ = 0."""

from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

from tracking.birth_policy import BirthHead, QBirthByteTrack
from tracking.decision_log import IndexedDetections
from tracking.offline_association import _DetSlice, make_tracker_args
from tracking.xywh_bytetrack import XYWHByteTrack

ARGS = dict(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=30, match_thresh=0.8)


def _frames(n: int = 6):
    frames = []
    for t in range(n):
        boxes = [[10 + 2 * t, 10, 40 + 2 * t, 40], [200 - t, 100, 260 - t, 160]]
        scores = [0.9, 0.6]
        if t >= 3:
            boxes.append([400, 300, 430, 330]); scores.append(0.3)  # новое рождение на кадре 3
        frames.append((np.asarray(boxes, np.float32), np.asarray(scores, np.float32), np.full(len(scores), 2, np.float32)))
    return frames


def _constant_head(value: float) -> BirthHead:
    """Голова, возвращающая одно и то же q_birth (для тестов политики)."""
    names = ("conf",)
    return BirthHead(feature_names=names, mean=np.zeros(1), scale=np.ones(1), coef=np.zeros(1), intercept=float(np.log(value / (1 - value))))


def _gray():
    return np.full((400, 600), 128, np.uint8)


def test_birth_head_is_logistic_on_standardised_features() -> None:
    head = BirthHead(feature_names=("a", "b"), mean=np.asarray([1.0, 2.0]), scale=np.asarray([2.0, 4.0]), coef=np.asarray([1.0, -1.0]), intercept=0.5)
    q = head.probability({"a": 3.0, "b": 2.0})  # z = (1)(1) + (-1)(0) + 0.5 = 1.5
    np.testing.assert_allclose(q, 1 / (1 + np.exp(-1.5)))


def test_tau_zero_reproduces_frozen_tracker_byte_for_byte() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    plain = XYWHByteTrack(make_tracker_args(**ARGS))
    expected = [np.array(plain.update(_DetSlice(*f)), copy=True) for f in frames]
    BaseTrack.reset_id()
    policy = QBirthByteTrack(make_tracker_args(**ARGS), head=_constant_head(0.5), tau=0.0)
    actual = []
    for f in frames:
        policy.set_frame_context(gray=_gray(), prev_propagated=np.zeros((0, 4), np.float32), frame_size=(600, 400))
        actual.append(np.array(policy.update(IndexedDetections(*f)), copy=True))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_tau_one_vetoes_every_birth() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    policy = QBirthByteTrack(make_tracker_args(**ARGS), head=_constant_head(0.5), tau=1.0)
    outputs = []
    for f in frames:
        policy.set_frame_context(gray=_gray(), prev_propagated=np.zeros((0, 4), np.float32), frame_size=(600, 400))
        outputs.append(np.array(policy.update(IndexedDetections(*f)), copy=True))
    ids = {int(row[4]) for out in outputs for row in out}
    assert ids == set()  # вето действует на всех кадрах, включая первый
    assert policy.birth_log and all(entry["vetoed"] for entry in policy.birth_log)


def test_birth_log_records_q_birth_for_every_candidate() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    policy = QBirthByteTrack(make_tracker_args(**ARGS), head=_constant_head(0.7), tau=0.6)
    for f in frames:
        policy.set_frame_context(gray=_gray(), prev_propagated=np.zeros((0, 4), np.float32), frame_size=(600, 400))
        policy.update(IndexedDetections(*f))
    candidates = [e for e in policy.birth_log if e["frame"] == 4]
    assert candidates and all(abs(e["q_birth"] - 0.7) < 1e-6 and not e["vetoed"] for e in candidates)
