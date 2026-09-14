"""Самостоятельный многoобъектный трекер с внешней моделью движения.

YOLO26 поставляет независимые обнаружения каждого кадра. ``FlyVisTracker``
поддерживает постоянные идентификаторы, жизненный цикл траекторий и решает
задачу назначения. FlyVis используется только через ``MotionField``: это
позволяет тем же каркасом позднее проверить Farneback, DIS и контроль без
оптического потока, не меняя логику трекера.

Координаты рамок имеют формат ``xyxy``. Выход ``update`` совместим с
Ultralytics-трекерами: ``x1,y1,x2,y2,track_id,score,class_id,detection_index``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment


_EPS = 1e-12


def _boxes(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32).reshape(-1, 4)
    if len(value) and (
        not np.isfinite(value).all()
        or np.any(value[:, 2] <= value[:, 0])
        or np.any(value[:, 3] <= value[:, 1])
    ):
        raise ValueError("Рамки должны быть конечными и иметь положительную площадь")
    return value


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Матрица IoU (пересечения относительно объединения) двух наборов рамок."""
    left = _boxes(left)
    right = _boxes(right)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0).prod(axis=2)
    left_area = (left[:, 2:] - left[:, :2]).prod(axis=1)[:, None]
    right_area = (right[:, 2:] - right[:, :2]).prod(axis=1)[None, :]
    return intersection / np.maximum(left_area + right_area - intersection, _EPS)


def _centers(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def _areas(boxes: np.ndarray) -> np.ndarray:
    return np.maximum((boxes[:, 2:] - boxes[:, :2]).prod(axis=1), _EPS)


def shift_box(box: np.ndarray, vector_xy: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_xy, dtype=np.float32).reshape(2)
    return np.asarray(box, dtype=np.float32) + np.tile(vector, 2)


@dataclass(frozen=True)
class MotionEstimate:
    """Движение объекта и диагностическое движение окружающего фона."""

    vector_xy: np.ndarray
    background_xy: np.ndarray
    relative_xy: np.ndarray
    confidence: float
    object_samples: int
    background_samples: int


@dataclass(frozen=True)
class MotionField:
    """Разреженное поле движения в координатах исходного изображения."""

    points_xy: np.ndarray
    vectors_xy: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xy, dtype=np.float32).reshape(-1, 2)
        vectors = np.asarray(self.vectors_xy, dtype=np.float32).reshape(-1, 2)
        if points.shape != vectors.shape:
            raise ValueError("Координаты и векторы поля должны иметь одинаковую форму")
        if not np.isfinite(points).all() or not np.isfinite(vectors).all():
            raise ValueError("Поле движения содержит NaN или Inf")
        object.__setattr__(self, "points_xy", points)
        object.__setattr__(self, "vectors_xy", vectors)

    @classmethod
    def empty(cls) -> "MotionField":
        return cls(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))

    def estimate_box(
        self,
        box: np.ndarray,
        *,
        context_scale: float = 2.0,
        min_samples: int = 3,
    ) -> MotionEstimate:
        """Робастно оценить движение внутри рамки и в окружающем кольце.

        Для прогноза используется полное движение внутри рамки. Движение фона
        и разность сохраняются отдельно: вычитать фон из прогноза нельзя,
        поскольку смещение камеры также перемещает рамку в координатах кадра.
        """
        if context_scale <= 1.0:
            raise ValueError("context_scale должен быть больше единицы")
        if min_samples < 1:
            raise ValueError("min_samples должен быть положительным")
        box = _boxes(np.asarray(box).reshape(1, 4))[0]
        zero = np.zeros(2, dtype=np.float32)
        if not len(self.points_xy):
            return MotionEstimate(zero, zero, zero, 0.0, 0, 0)

        center = (box[:2] + box[2:]) * 0.5
        half = (box[2:] - box[:2]) * 0.5
        inside = np.all(
            (self.points_xy >= box[:2]) & (self.points_xy <= box[2:]), axis=1
        )
        object_indices = np.flatnonzero(inside)
        if len(object_indices) < min_samples:
            normalized = (self.points_xy - center) / np.maximum(half, 1.0)
            nearest = np.argsort(np.linalg.norm(normalized, axis=1))[:min_samples]
            object_indices = np.unique(np.concatenate([object_indices, nearest]))

        outer_min = center - half * context_scale
        outer_max = center + half * context_scale
        in_outer = np.all(
            (self.points_xy >= outer_min) & (self.points_xy <= outer_max), axis=1
        )
        background_indices = np.flatnonzero(in_outer & ~inside)

        object_vectors = self.vectors_xy[object_indices]
        object_vector = np.median(object_vectors, axis=0).astype(np.float32)
        if len(background_indices):
            background = np.median(
                self.vectors_xy[background_indices], axis=0
            ).astype(np.float32)
        else:
            background = zero.copy()
        relative = object_vector - background

        deviations = np.linalg.norm(object_vectors - object_vector, axis=1)
        dispersion = float(np.median(deviations)) if len(deviations) else np.inf
        coverage = min(1.0, len(object_indices) / float(min_samples))
        confidence = float(coverage / (1.0 + dispersion))
        return MotionEstimate(
            vector_xy=object_vector,
            background_xy=background,
            relative_xy=relative,
            confidence=confidence,
            object_samples=int(len(object_indices)),
            background_samples=int(len(background_indices)),
        )


class MotionFieldEstimator(Protocol):
    """Интерфейс источника движения для одного видеопотока."""

    def reset(self) -> None: ...

    def observe(self, frame: np.ndarray) -> MotionField: ...


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    class_id: int
    score: float
    status: TrackStatus
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    velocity_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    motion_confidence: float = 0.0
    last_detection_index: int = -1

    def __post_init__(self) -> None:
        self.box = _boxes(np.asarray(self.box).reshape(1, 4))[0].copy()
        self.velocity_xy = np.asarray(self.velocity_xy, dtype=np.float32).reshape(2)


@dataclass(frozen=True)
class FlyVisTrackerConfig:
    detection_threshold: float = 0.25
    new_track_threshold: float = 0.25
    min_hits: int = 1
    max_age: int = 30
    iou_weight: float = 0.35
    motion_weight: float = 0.55
    size_weight: float = 0.10
    max_motion_distance: float = 2.5
    min_iou_gate: float = 0.01
    max_size_log_change: float = 1.25
    max_assignment_cost: float = 2.0
    context_scale: float = 2.0
    min_motion_samples: int = 3
    reuse_last_motion: bool = True
    update_velocity_from_matches: bool = False

    def __post_init__(self) -> None:
        if self.min_hits < 1 or self.max_age < 0:
            raise ValueError("min_hits должен быть >= 1, max_age должен быть >= 0")
        weights = self.iou_weight + self.motion_weight + self.size_weight
        if not np.isclose(weights, 1.0):
            raise ValueError("Веса iou, motion и size должны суммироваться в 1")
        if self.new_track_threshold < self.detection_threshold:
            raise ValueError("new_track_threshold не может быть ниже detection_threshold")


class MotionFusionPolicy(Protocol):
    """Правило превращения сырой оценки движения в применяемую скорость."""

    def fuse(
        self,
        estimate: MotionEstimate,
        previous_velocity: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[np.ndarray, float]: ...


class HeuristicMotionFusion:
    """Текущее правило смешивания, выделенное как заменяемый контроль.

    Возвращаемое число является весом внутреннего согласия поля, а не
    откалиброванной вероятностью правильного движения.
    """

    def fuse(
        self,
        estimate: MotionEstimate,
        previous_velocity: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[np.ndarray, float]:
        weight = float(np.clip(estimate.confidence, 0.0, 1.0))
        if config.reuse_last_motion:
            velocity = (
                weight * estimate.vector_xy
                + (1.0 - weight) * previous_velocity
            )
        else:
            velocity = weight * estimate.vector_xy
        return np.asarray(velocity, dtype=np.float32), weight


class AssignmentPolicy(Protocol):
    """Интерфейс стоимости и назначения пар «трек — детекция»."""

    def cost(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def assign(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]: ...


class WeightedGeometryAssignment:
    """Текущая взвешенная геометрическая ассоциация как явный контроль."""

    def cost(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(tracks) or not len(detections):
            shape = (len(tracks), len(detections))
            return np.zeros(shape, np.float32), np.zeros(shape, dtype=bool)

        iou = box_iou(predicted, detections)
        track_centers = _centers(predicted)
        detection_centers = _centers(detections)
        scale = np.sqrt(_areas(predicted))[:, None]
        motion_distance = np.linalg.norm(
            track_centers[:, None, :] - detection_centers[None, :, :], axis=2
        ) / np.maximum(scale, 1.0)

        track_wh = np.maximum(predicted[:, 2:] - predicted[:, :2], 1.0)
        detection_wh = np.maximum(detections[:, 2:] - detections[:, :2], 1.0)
        size_distance = np.mean(
            np.abs(np.log(detection_wh[None, :, :] / track_wh[:, None, :])), axis=2
        )
        track_classes = np.asarray([track.class_id for track in tracks], dtype=np.int64)
        same_class = track_classes[:, None] == classes[None, :]
        valid = (
            same_class
            & (size_distance <= config.max_size_log_change)
            & (
                (motion_distance <= config.max_motion_distance)
                | (iou >= config.min_iou_gate)
            )
        )
        cost = (
            config.iou_weight * (1.0 - iou)
            + config.motion_weight * motion_distance
            + config.size_weight * size_distance
        ).astype(np.float32)
        cost[~valid] = np.inf
        return cost, valid

    def assign(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
        config: FlyVisTrackerConfig,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        if not len(tracks) or not len(detections):
            return [], unmatched_tracks, unmatched_detections
        cost, valid = self.cost(tracks, predicted, detections, classes, config)
        finite_cost = np.where(np.isfinite(cost), cost, 1e6)
        rows, cols = linear_sum_assignment(finite_cost)
        matches = []
        for row, col in zip(rows.tolist(), cols.tolist()):
            if not valid[row, col] or cost[row, col] > config.max_assignment_cost:
                continue
            matches.append((row, col))
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)
        return matches, unmatched_tracks, unmatched_detections


class FlyVisTracker:
    """Независимый трекер: FlyVis-движение, назначение и жизненный цикл ID."""

    def __init__(
        self,
        config: FlyVisTrackerConfig | None = None,
        motion_estimator: MotionFieldEstimator | None = None,
        motion_fusion: MotionFusionPolicy | None = None,
        assignment: AssignmentPolicy | None = None,
    ) -> None:
        self.config = config or FlyVisTrackerConfig()
        self.motion_estimator = motion_estimator
        self.motion_fusion = motion_fusion or HeuristicMotionFusion()
        self.assignment = assignment or WeightedGeometryAssignment()
        self.reset()

    def reset(self) -> None:
        self.frame_index = -1
        self.next_track_id = 1
        self.tracks: list[Track] = []
        self.events: list[dict] = []
        self.motion_history: list[dict] = []
        if self.motion_estimator is not None:
            self.motion_estimator.reset()

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.status != TrackStatus.REMOVED]

    def _predict(self, field: MotionField) -> tuple[list[Track], np.ndarray]:
        tracks = self.active_tracks
        predicted = []
        for track in tracks:
            estimate = field.estimate_box(
                track.box,
                context_scale=self.config.context_scale,
                min_samples=self.config.min_motion_samples,
            )
            velocity, reliability_weight = self.motion_fusion.fuse(
                estimate, track.velocity_xy, self.config
            )
            track.velocity_xy = velocity
            track.motion_confidence = reliability_weight
            self.motion_history.append(
                {
                    "frame": self.frame_index,
                    "track_id": track.track_id,
                    "dx": float(track.velocity_xy[0]),
                    "dy": float(track.velocity_xy[1]),
                    "confidence": reliability_weight,
                    "object_samples": estimate.object_samples,
                    "background_samples": estimate.background_samples,
                }
            )
            predicted.append(shift_box(track.box, track.velocity_xy))
        if not predicted:
            return tracks, np.zeros((0, 4), dtype=np.float32)
        return tracks, np.stack(predicted).astype(np.float32)

    def _cost(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.assignment.cost(
            tracks, predicted, detections, classes, self.config
        )

    def _assign(
        self,
        tracks: list[Track],
        predicted: np.ndarray,
        detections: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        return self.assignment.assign(
            tracks, predicted, detections, classes, self.config
        )

    def update(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        *,
        frame: np.ndarray | None = None,
        motion_field: MotionField | None = None,
    ) -> np.ndarray:
        """Обработать кадр и вернуть только подтверждённые найденные траектории."""
        self.frame_index += 1
        detections_all = _boxes(xyxy)
        confidence_all = np.asarray(confidence, dtype=np.float32).reshape(-1)
        classes_all = np.asarray(class_id, dtype=np.int64).reshape(-1)
        if not (
            len(detections_all) == len(confidence_all) == len(classes_all)
        ):
            raise ValueError("Число рамок, оценок уверенности и классов не совпало")
        if not np.isfinite(confidence_all).all():
            raise ValueError("Оценки уверенности содержат NaN или Inf")

        keep = confidence_all >= self.config.detection_threshold
        original_indices = np.flatnonzero(keep)
        detections = detections_all[keep]
        scores = confidence_all[keep]
        classes = classes_all[keep]

        if motion_field is not None and frame is not None:
            raise ValueError("Укажите frame или motion_field, но не оба сразу")
        if motion_field is None:
            if self.motion_estimator is None:
                motion_field = MotionField.empty()
            else:
                if frame is None:
                    raise ValueError("Для подключённого оценщика движения нужен frame")
                motion_field = self.motion_estimator.observe(frame)

        tracks, predicted = self._predict(motion_field)
        matches, unmatched_tracks, unmatched_detections = self._assign(
            tracks, predicted, detections, classes
        )
        visible: list[Track] = []

        for track_index, detection_index in matches:
            track = tracks[track_index]
            previous_status = track.status
            previous_center = (track.box[:2] + track.box[2:]) * 0.5
            detection_center = (
                detections[detection_index, :2] + detections[detection_index, 2:]
            ) * 0.5
            track.box = detections[detection_index].copy()
            track.class_id = int(classes[detection_index])
            track.score = float(scores[detection_index])
            track.hits += 1
            track.age += 1
            track.time_since_update = 0
            track.last_detection_index = int(original_indices[detection_index])
            if self.config.update_velocity_from_matches:
                track.velocity_xy = (detection_center - previous_center).astype(np.float32)
                track.motion_confidence = 1.0
            if track.hits >= self.config.min_hits:
                track.status = TrackStatus.CONFIRMED
            event = "refound" if previous_status == TrackStatus.LOST else "matched"
            self.events.append(
                {"frame": self.frame_index, "event": event, "track_id": track.track_id}
            )
            if track.status == TrackStatus.CONFIRMED:
                visible.append(track)

        for track_index in sorted(unmatched_tracks):
            track = tracks[track_index]
            previous_status = track.status
            track.box = predicted[track_index].copy()
            track.age += 1
            track.time_since_update += 1
            if track.status == TrackStatus.TENTATIVE:
                track.status = TrackStatus.REMOVED
                event = "removed_tentative"
            elif track.time_since_update > self.config.max_age:
                track.status = TrackStatus.REMOVED
                event = "removed"
            else:
                track.status = TrackStatus.LOST
                event = "missed_lost" if previous_status == TrackStatus.LOST else "lost"
            self.events.append(
                {"frame": self.frame_index, "event": event, "track_id": track.track_id}
            )

        for detection_index in sorted(unmatched_detections):
            if scores[detection_index] < self.config.new_track_threshold:
                continue
            status = (
                TrackStatus.CONFIRMED
                if self.config.min_hits <= 1
                else TrackStatus.TENTATIVE
            )
            track = Track(
                track_id=self.next_track_id,
                box=detections[detection_index],
                class_id=int(classes[detection_index]),
                score=float(scores[detection_index]),
                status=status,
                last_detection_index=int(original_indices[detection_index]),
            )
            self.next_track_id += 1
            self.tracks.append(track)
            self.events.append(
                {"frame": self.frame_index, "event": "created", "track_id": track.track_id}
            )
            if track.status == TrackStatus.CONFIRMED:
                visible.append(track)

        visible.sort(key=lambda track: track.track_id)
        output = np.zeros((len(visible), 8), dtype=np.float32)
        for row_index, track in enumerate(visible):
            output[row_index, :4] = track.box
            output[row_index, 4] = track.track_id
            output[row_index, 5] = track.score
            output[row_index, 6] = track.class_id
            output[row_index, 7] = track.last_detection_index
        return output
