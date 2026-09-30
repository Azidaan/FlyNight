"""Сквозной трекер для проверки сигналов надёжности предсказания (ночная работа 30.09.2026).

`MotionSourceByteTrack` — надстройка над `XYWHByteTrack` с двумя рычагами, на которые
подаётся надёжность предсказания фильтра (голова без мухи или с мухой):

* источник движения: центр рамки из фильтра Калмана или перенос последней рамки
  трека по кэшированному полю DIS на один кадр; жёстко (`dis`) или по порогу
  вероятности ненадёжности (`switch`);
* буфер IoU при первой ассоциации: рамки трека и детекции расширяются на долю b
  своей стороны (заимствовано: C-BIoU, Yang et al., WACV 2023 «Hard to Track Objects
  with Irregular Motions and Similar Appearances»); b фиксирован или b = b0 + k·p.

При `motion="kalman"`, `buffer=0` и без головы поведение байт-в-байт равно стенду
`XYWHByteTrack` (тест). Разметку трекер не читает; поле DIS и надёжность передаются
снаружи по кадру.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from ultralytics.trackers.utils import matching

from tracking.xywh_bytetrack import XYWHByteTrack

ReliabilityFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]  # (предсказания xywh, DIS xyxy, номер кадра) -> p(ненадёжно)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xywh = np.asarray(xywh, dtype=np.float64)
    half = xywh[..., 2:4] / 2.0
    return np.concatenate([xywh[..., :2] - half, xywh[..., :2] + half], axis=-1)


def buffered(xyxy: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """Расширить рамки (N, 4) на долю b ширины/высоты с каждой стороны."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64).reshape(-1), (len(xyxy),))
    wh = np.maximum(xyxy[:, 2:] - xyxy[:, :2], 1.0)
    pad = wh * b[:, None]
    return np.concatenate([xyxy[:, :2] - pad, xyxy[:, 2:] + pad], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class MotionSourceByteTrack(XYWHByteTrack):
    def __init__(self, args, *, motion: str = "kalman", buffer: float = 0.0,
                 buffer_gain: float = 0.0, switch_threshold: float = 0.5,
                 dis_propagate: Callable[[np.ndarray, int], np.ndarray] | None = None,
                 reliability: ReliabilityFn | None = None) -> None:
        super().__init__(args)
        if motion not in {"kalman", "dis", "switch"}:
            raise ValueError("motion должен быть kalman, dis или switch")
        if motion != "kalman" and dis_propagate is None:
            raise ValueError("для motion=dis/switch нужен dis_propagate")
        if (motion == "switch" or buffer_gain) and reliability is None:
            raise ValueError("для switch и адаптивного буфера нужна голова надёжности")
        self.motion = motion
        self.buffer = float(buffer)
        self.buffer_gain = float(buffer_gain)
        self.switch_threshold = float(switch_threshold)
        self.dis_propagate = dis_propagate
        self.reliability = reliability
        self._previous_xywh: dict[int, np.ndarray] = {}
        self._p_bad: dict[int, float] = {}
        self.stats = {"switched": 0, "tracks_seen": 0}

    # Предсказание: запомнить рамку до шага фильтра, чтобы перенести её по полю DIS.
    def multi_predict(self, tracks) -> None:
        self._previous_xywh = {id(t): np.asarray(t.mean[:4], dtype=np.float64).copy() for t in tracks if t.mean is not None}
        super().multi_predict(tracks)

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high) -> None:
        super()._pre_first_associate(strack_pool, unconfirmed, img, results_high)
        self._p_bad = {}
        tracks = [t for t in strack_pool if id(t) in self._previous_xywh]
        if not tracks or (self.motion == "kalman" and not self.buffer_gain):
            return
        raw_index = self.frame_id - 1
        pred_xywh = np.stack([np.asarray(t.mean[:4], dtype=np.float64) for t in tracks])
        prev_xyxy = xywh_to_xyxy(np.stack([self._previous_xywh[id(t)] for t in tracks]))
        dis_xyxy = np.stack([self.dis_propagate(box, raw_index) for box in prev_xyxy]) if self.dis_propagate is not None else None
        p_bad = self.reliability(pred_xywh, dis_xyxy, raw_index) if self.reliability is not None else None
        self.stats["tracks_seen"] += len(tracks)
        for k, track in enumerate(tracks):
            if p_bad is not None:
                self._p_bad[id(track)] = float(p_bad[k])
            use_dis = self.motion == "dis" or (self.motion == "switch" and p_bad[k] > self.switch_threshold)
            if use_dis and np.isfinite(dis_xyxy[k]).all():
                center = (dis_xyxy[k, :2] + dis_xyxy[k, 2:]) / 2.0
                track.mean[:2] = center
                self.stats["switched"] += 1

    # Первая ассоциация: буферизованный IoU (C-BIoU), буфер трека — фиксированный или по надёжности.
    def get_dists(self, tracks, detections):
        if self.buffer <= 0 and not self.buffer_gain:
            return super().get_dists(tracks, detections)
        track_boxes = np.asarray([t.xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)
        det_boxes = np.asarray([d.xyxy for d in detections], dtype=np.float64).reshape(-1, 4)
        b_track = np.asarray([self.buffer + self.buffer_gain * self._p_bad.get(id(t), 0.0) for t in tracks])
        if not self.buffer_gain:
            iou = iou_matrix(buffered(track_boxes, self.buffer), buffered(det_boxes, self.buffer))
        else:
            # В каждой паре обе рамки расширяются на буфер трека (как в C-BIoU); при разных буферах
            # идеально совпадающая пара получила бы IoU < 1.
            iou = np.zeros((len(track_boxes), len(det_boxes)))
            for i in range(len(track_boxes)):
                iou[i] = iou_matrix(buffered(track_boxes[i : i + 1], b_track[i]), buffered(det_boxes, b_track[i]))[0]
        dists = 1.0 - iou
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists


class BufferedIoUMixin:
    """Статический буфер IoU (C-BIoU, заимствовано) для любого наследника BYTETracker.

    Ставится первым в списке базовых классов; `self.buffer` задаёт наследник.
    """

    buffer: float = 0.0

    def get_dists(self, tracks, detections):
        if self.buffer <= 0:
            return super().get_dists(tracks, detections)
        track_boxes = np.asarray([t.xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)
        det_boxes = np.asarray([d.xyxy for d in detections], dtype=np.float64).reshape(-1, 4)
        dists = 1.0 - iou_matrix(buffered(track_boxes, self.buffer), buffered(det_boxes, self.buffer))
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists


def make_deferred_buffered(args, *, buffer: float, **kwargs):
    """Отложенное рождение по q_birth (tracking.birth_policy) + буфер IoU."""
    from tracking.birth_policy import DeferredBirthByteTrack

    class DeferredBufferedByteTrack(BufferedIoUMixin, DeferredBirthByteTrack):
        pass

    tracker = DeferredBufferedByteTrack(args, **kwargs)
    tracker.buffer = float(buffer)
    return tracker
