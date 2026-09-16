"""Правила меток и признаков для выборок событийной уверенности (карточка event_dataset_v1).

Метки строятся постфактум из разметки опорных кадров; признаки — только из
того, что доступно трекеру на инференсе. Модуль не читает файлы.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import numpy as np

from tracking.flyvis_tracker import box_iou


class IdentityTimeline:
    """GT-идентичность трека по опорным кадрам: последняя размеченная детекция, с которой трек совпал."""

    def __init__(self) -> None:
        self._refs: dict[int, list[int]] = defaultdict(list)
        self._ids: dict[int, list[str]] = defaultdict(list)

    def record(self, *, ref_index: int, track_id: int, gt_id: str | None) -> None:
        if gt_id is None:
            return
        refs = self._refs[track_id]
        if refs and ref_index < refs[-1]:
            raise ValueError("опорные кадры должны записываться по возрастанию")
        refs.append(int(ref_index))
        self._ids[track_id].append(str(gt_id))

    def before(self, track_id: int, ref_index: int) -> str | None:
        refs = self._refs.get(track_id)
        if not refs:
            return None
        position = bisect.bisect_left(refs, ref_index)
        return self._ids[track_id][position - 1] if position else None

    def upto(self, track_id: int, ref_index: int) -> str | None:
        refs = self._refs.get(track_id)
        if not refs:
            return None
        position = bisect.bisect_right(refs, ref_index)
        return self._ids[track_id][position - 1] if position else None


def birth_label(*, first_ref: int, gt_id: str | None, presence: dict[str, set[int]], horizon: int, n_ref: int) -> int | None:
    """1 — объект есть на first_ref и на всех следующих horizon−1 опорных кадрах; None — цензура."""
    last = first_ref + horizon - 1
    if last >= n_ref:
        return None
    if gt_id is None:
        return 0
    frames = presence.get(gt_id, set())
    return int(all(index in frames for index in range(first_ref, last + 1)))


def alive_label(*, lost_after_ref: int, gt_id: str, presence: dict[str, set[int]], window: int, n_ref: int) -> int | None:
    """1 — объект появляется в GT на опорных кадрах (lost_after_ref, lost_after_ref + window]; None — цензура."""
    last = lost_after_ref + window
    if last >= n_ref:
        return None
    frames = presence.get(gt_id, set())
    return int(any(index in frames for index in range(lost_after_ref + 1, last + 1)))


def admissible_candidates(predictions: np.ndarray, prediction_classes: np.ndarray, detections: np.ndarray, detection_classes: np.ndarray, *, distance_scale: float = 2.5) -> list[list[int]]:
    """Для каждого предсказания — индексы детекций того же класса с IoU > 0 или близким центром."""
    predictions = np.asarray(predictions, np.float32).reshape(-1, 4)
    detections = np.asarray(detections, np.float32).reshape(-1, 4)
    if not len(predictions) or not len(detections):
        return [[] for _ in range(len(predictions))]
    overlaps = box_iou(predictions, detections)
    p_center = (predictions[:, :2] + predictions[:, 2:]) / 2
    d_center = (detections[:, :2] + detections[:, 2:]) / 2
    scale = np.sqrt(np.maximum((predictions[:, 2] - predictions[:, 0]) * (predictions[:, 3] - predictions[:, 1]), 1.0))
    distance = np.linalg.norm(p_center[:, None, :] - d_center[None, :, :], axis=-1) / scale[:, None]
    same_class = np.asarray(prediction_classes)[:, None] == np.asarray(detection_classes)[None, :]
    admissible = same_class & ((overlaps > 0.0) | (distance < distance_scale))
    return [np.flatnonzero(row).tolist() for row in admissible]


def image_box_stats(gray: np.ndarray, box, *, dark: int = 26, bright: int = 230) -> dict[str, float]:
    """Средняя яркость, контраст и доли тёмных/почти белых пикселей внутри рамки."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return {"mean": 0.0, "std": 0.0, "dark_fraction": 1.0, "bright_fraction": 0.0}
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    return {
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "dark_fraction": float((patch < dark).mean()),
        "bright_fraction": float((patch > bright).mean()),
    }


def frames_to_exit(box, *, velocity, frame_size) -> float:
    """Через сколько кадров рамка при постоянной скорости коснётся края кадра (начнёт выходить)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = frame_size
    vx, vy = (float(v) for v in velocity)
    candidates = []
    if vx > 0:
        candidates.append(max(width - x2, 0.0) / vx)
    elif vx < 0:
        candidates.append(max(x1, 0.0) / -vx)
    if vy > 0:
        candidates.append(max(height - y2, 0.0) / vy)
    elif vy < 0:
        candidates.append(max(y1, 0.0) / -vy)
    return float(min(candidates)) if candidates else float(np.inf)


def edge_distance(box, frame_size) -> float:
    """Расстояние от рамки до ближайшего края кадра в долях меньшей стороны кадра."""
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = frame_size
    return float(min(x1, y1, width - x2, height - y2) / max(min(width, height), 1.0))
