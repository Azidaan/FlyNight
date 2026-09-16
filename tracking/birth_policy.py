"""Политика рождения по q_birth поверх неизменённого XYWHByteTrack (карточка birth_policy_qbirth_v1).

Единственное отличие от стенда — правило рождения нового ID: среди кандидатов
со score ≥ new_track_thresh трек создаётся только при q_birth ≥ τ. Оба прохода
ассоциации, пороги и буфер не трогаются. Признаки q_birth — те же функции, что
в build_event_dataset.py; разметка не используется.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics.trackers.basetrack import TrackState

from tracking.decision_log import AuditedXYWHByteTrack
from tracking.flyvis_tracker import box_iou


@dataclass(frozen=True)
class BirthHead:
    """Логистическая голова q_birth: стандартизация + линейная модель, сохранённая из sklearn."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def probability(self, features: dict[str, float]) -> float:
        x = np.asarray([features[name] for name in self.feature_names], np.float64)
        z = float(((x - self.mean) / self.scale) @ self.coef + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))

    @classmethod
    def from_pipeline(cls, pipeline, feature_names) -> "BirthHead":
        scaler, model = pipeline.steps[0][1], pipeline.steps[1][1]
        return cls(tuple(feature_names), np.asarray(scaler.mean_, np.float64), np.asarray(scaler.scale_, np.float64), np.asarray(model.coef_[0], np.float64), float(model.intercept_[0]))

    def to_json(self) -> dict:
        return {"feature_names": list(self.feature_names), "mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_json(cls, data: dict | str | Path) -> "BirthHead":
        if not isinstance(data, dict):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        return cls(tuple(data["feature_names"]), np.asarray(data["mean"], np.float64), np.asarray(data["scale"], np.float64), np.asarray(data["coef"], np.float64), float(data["intercept"]))


def birth_features(*, box_xyxy, score: float, class_id: int, class_ids, gray, frame_size, tracked_boxes: np.ndarray, prev_propagated: np.ndarray) -> dict[str, float]:
    """Признаки рождения — те же, что DET/BIRTH_FEATURES в build_event_dataset.py."""
    from tracking.event_dataset import edge_distance, image_box_stats

    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    width, height = frame_size
    stats = image_box_stats(gray, box_xyxy)
    box = np.asarray(box_xyxy, np.float32)[None, :]
    if len(tracked_boxes):
        overlaps = box_iou(box, tracked_boxes)[0]
        ctx_max, ctx_near = float(overlaps.max()), float((overlaps > 0.3).sum())
    else:
        ctx_max, ctx_near = 0.0, 0.0
    if len(prev_propagated):
        overlaps = box_iou(box, prev_propagated)[0]
        dis_max, dis_exists = float(overlaps.max()), float(overlaps.max() >= 0.5)
    else:
        dis_max, dis_exists = 0.0, 0.0
    features = {"conf": float(score)}
    for c in class_ids:
        features[f"class_{c}"] = float(class_id == c)
    features.update({
        "width": (x2 - x1) / width, "height": (y2 - y1) / height, "area_fraction": (x2 - x1) * (y2 - y1) / (width * height),
        "center_x": (x1 + x2) / 2 / width, "center_y": (y1 + y2) / 2 / height, "edge_distance": edge_distance(box_xyxy, frame_size),
        "img_mean": stats["mean"] / 255.0, "img_std": stats["std"] / 255.0, "img_dark": stats["dark_fraction"], "img_bright": stats["bright_fraction"],
        "ctx_max_iou_tracked": ctx_max, "ctx_tracks_near": ctx_near, "dis_prev_max_iou": dis_max, "dis_prev_exists": dis_exists,
    })
    return features


class QBirthByteTrack(AuditedXYWHByteTrack):
    """XYWHByteTrack, где рождение разрешено только при q_birth ≥ τ."""

    def __init__(self, args, *, head: BirthHead, tau: float, class_ids=(0, 1, 2, 3, 4, 5, 6, 7)) -> None:
        super().__init__(args)
        self.head = head
        self.tau = float(tau)
        self.class_ids = tuple(class_ids)
        self.birth_log: list[dict] = []
        self._context: dict | None = None

    def set_frame_context(self, *, gray, prev_propagated, frame_size) -> None:
        """Контекст кадра t для признаков рождения; вызывается перед update()."""
        self._context = {"gray": gray, "prev_propagated": np.asarray(prev_propagated, np.float32).reshape(-1, 4), "frame_size": tuple(frame_size)}

    def _tracked_prediction_boxes(self) -> np.ndarray:
        boxes = []
        for p in self.predictions:
            if p["state"] == "tracked":
                cx, cy, w, h = p["xywh"]
                w, h = max(w, 1.0), max(h, 1.0)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return np.asarray(boxes, np.float32).reshape(-1, 4)

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        if self._context is None:
            raise RuntimeError("set_frame_context() должен вызываться перед update()")
        if self.tau <= 0.0:
            return super()._init_new_tracks(u_detection, detections, activated, refind)
        tracked_boxes = self._tracked_prediction_boxes()
        allowed = []
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            features = birth_features(box_xyxy=track.xyxy, score=float(track.score), class_id=int(track.cls), class_ids=self.class_ids, gray=self._context["gray"], frame_size=self._context["frame_size"], tracked_boxes=tracked_boxes, prev_propagated=self._context["prev_propagated"])
            q = self.head.probability(features)
            vetoed = q < self.tau
            self.birth_log.append({"frame": int(self.frame_id), "det_index": int(track.idx), "score": float(track.score), "q_birth": q, "vetoed": bool(vetoed)})
            if not vetoed:
                allowed.append(inew)
        return super()._init_new_tracks(allowed, detections, activated, refind)
