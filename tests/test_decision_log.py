"""Цель 4: журнал решений XYWHByteTrack не меняет решений и записывает события."""

from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

from tracking.decision_log import AuditedXYWHByteTrack, IndexedDetections
from tracking.offline_association import _DetSlice, make_tracker_args
from tracking.xywh_bytetrack import XYWHByteTrack

ARGS = dict(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=30, match_thresh=0.8)


def _frames() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Два объекта; второй исчезает на кадрах 4–5 и возвращается; слабая детекция на кадре 3."""
    rng = np.random.default_rng(0)
    frames = []
    for t in range(8):
        boxes, scores = [[10 + 2 * t, 10, 40 + 2 * t, 40]], [0.9]
        if t not in (4, 5):
            boxes.append([200 - t, 100, 260 - t, 160])
            scores.append(0.85 if t != 3 else 0.2)
        boxes.append([500 + rng.integers(-3, 3), 300, 520, 320])  # ложная слабая рамка
        scores.append(0.15)
        frames.append((np.asarray(boxes, np.float32), np.asarray(scores, np.float32), np.full(len(scores), 2, np.float32)))
    return frames


def test_audited_tracker_outputs_are_identical_to_frozen_tracker() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    plain = XYWHByteTrack(make_tracker_args(**ARGS))
    expected = [np.array(plain.update(_DetSlice(*f)), copy=True) for f in frames]
    BaseTrack.reset_id()
    audited = AuditedXYWHByteTrack(make_tracker_args(**ARGS))
    actual = [np.array(audited.update(IndexedDetections(*f)), copy=True) for f in frames]
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_audited_tracker_records_predictions_before_update_and_events() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    tracker = AuditedXYWHByteTrack(make_tracker_args(**ARGS))
    logs = [tracker.update_with_log(IndexedDetections(*f)) for f in frames]
    first = logs[0]
    assert first["predictions"] == []  # до первого кадра треков нет
    assert {e["event"] for e in first["events"]} >= {"created"}
    created = [e for e in first["events"] if e["event"] == "created"]
    assert sorted(e["det_index"] for e in created) == [0, 1]  # слабая рамка (0.15) не рождает трек
    second = logs[1]
    ids = {p["track_id"] for p in second["predictions"]}
    assert len(ids) == 2
    for p in second["predictions"]:
        assert set(p) >= {"track_id", "xywh", "state", "age", "hits", "since_update", "velocity"}
        assert p["since_update"] == 1
    assert any(e["event"] == "continued" and e["det_index"] == 0 for e in second["events"])


def test_audited_tracker_reports_lost_transition_and_return() -> None:
    frames = _frames()
    BaseTrack.reset_id()
    tracker = AuditedXYWHByteTrack(make_tracker_args(**ARGS))
    logs = [tracker.update_with_log(IndexedDetections(*f)) for f in frames]
    lost = [(i, e) for i, log in enumerate(logs) for e in log["events"] if e["event"] == "state_transition" and e["to"] == "lost"]
    # слабая детекция 0,2 на кадре 3 отвергается вторым проходом (штраф IoU·conf) — трек теряется уже там
    assert lost and lost[0][0] == 3
    lost_id = lost[0][1]["track_id"]
    later = [e for log in logs[6:] for e in log["events"] if e["event"] in ("reactivated", "continued") and e["track_id"] == lost_id]
    assert later, "трек должен вернуться после пропуска"
    # предсказание потерянного трека на кадре 5 хранит состояние lost и since_update > 1
    lost_pred = [p for p in logs[5]["predictions"] if p["track_id"] == lost_id]
    assert lost_pred and lost_pred[0]["state"] == "lost" and lost_pred[0]["since_update"] >= 2
