"""Голова надёжности предсказания для трекера (карточка e2e_motion_source_v1).

p(ненадёжно) = σ(w·z(x) + b) — логистическая регрессия, обученная скриптом
`scripts/train_reliability_head.py` на обучающих видео. Признаки считаются онлайн:

* геометрия предсказанной рамки (как в Н2): w/W, h/H, площадь, центр, сторона вырезки глаза;
* рассогласование DIS: |центр переноса DIS − центр Калмана| / sqrt(площади предсказания);
* (вариант с мухой) остаточный поток фовеального глаза: |δ|, δx, δy, средний |поток| в рамке,
  доля колонок внутри рамки — из кэша фовеи того же ролика; строка кэша сопоставляется с треком
  по кадру и IoU предсказанной рамки (≥ fly_min_iou). Нет пары — используется голова без мухи.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FRAME_W, FRAME_H = 1280.0, 720.0
BASE_FEATURES = ("box_w", "box_h", "box_area", "center_x", "center_y", "crop_side", "dis_disagreement", "dis_missing")
FLY_FEATURES = ("fly_norm", "fly_dx", "fly_dy", "fly_magnitude", "fly_n_inside")


def base_features(pred_xywh: np.ndarray, dis_xyxy: np.ndarray | None) -> np.ndarray:
    pred_xywh = np.asarray(pred_xywh, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = pred_xywh.T
    w = np.maximum(w, 1.0); h = np.maximum(h, 1.0)
    side = np.minimum(np.maximum(2.0 * np.maximum(w, h), 64.0), FRAME_H)
    if dis_xyxy is None:
        dis_xyxy = np.full((len(pred_xywh), 4), np.nan)
    dis_xyxy = np.asarray(dis_xyxy, dtype=np.float64).reshape(-1, 4)
    dis_center = (dis_xyxy[:, :2] + dis_xyxy[:, 2:]) / 2.0
    missing = ~np.isfinite(dis_center).all(axis=1)
    disagreement = np.linalg.norm(dis_center - np.stack([cx, cy], axis=1), axis=1) / np.sqrt(w * h)
    disagreement = np.where(missing, 0.0, disagreement)
    return np.stack([w / FRAME_W, h / FRAME_H, w * h / (FRAME_W * FRAME_H), cx / FRAME_W, cy / FRAME_H, side / FRAME_W, disagreement, missing.astype(np.float64)], axis=1)


def fly_features(delta: np.ndarray, magnitude: np.ndarray, n_inside: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).reshape(-1, 2)
    return np.stack([np.linalg.norm(delta, axis=1), delta[:, 0], delta[:, 1], np.asarray(magnitude, dtype=np.float64), np.asarray(n_inside, dtype=np.float64) / 721.0], axis=1)


class LogisticHead:
    def __init__(self, spec: dict) -> None:
        self.features = list(spec["features"])
        self.mean = np.asarray(spec["mean"], dtype=np.float64)
        self.scale = np.asarray(spec["scale"], dtype=np.float64)
        self.coef = np.asarray(spec["coef"], dtype=np.float64)
        self.intercept = float(spec["intercept"])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))


def xywh_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2] - b[:, 2:] / 2.0, b[:, :2] + b[:, 2:] / 2.0], axis=1)


class FlyLookup:
    """Строки кэша фовеи ролика по кадру: pred_box, delta, magnitude, n_inside."""

    def __init__(self, npz_path: Path, min_iou: float = 0.8) -> None:
        with np.load(npz_path) as z:
            frame = z["frame"]; order = np.argsort(frame, kind="stable")
            self.frame = frame[order]; self.pred_box = z["pred_box"][order].astype(np.float64)
            self.delta = z["delta"][order]; self.magnitude = z["magnitude"][order]; self.n_inside = z["n_inside"][order]
        self.min_iou = float(min_iou)
        self.stats = {"queries": 0, "matched": 0}

    def query(self, raw_index: int, pred_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from tracking.motion_source_bytetrack import iou_matrix

        a, b = np.searchsorted(self.frame, raw_index, "left"), np.searchsorted(self.frame, raw_index, "right")
        boxes = xywh_xyxy(pred_xywh)
        feats = np.zeros((len(boxes), len(FLY_FEATURES))); ok = np.zeros(len(boxes), dtype=bool)
        self.stats["queries"] += len(boxes)
        if b > a and len(boxes):
            iou = iou_matrix(boxes, self.pred_box[a:b])
            best = iou.argmax(axis=1); good = iou[np.arange(len(boxes)), best] >= self.min_iou
            rows = a + best[good]
            feats[good] = fly_features(self.delta[rows], self.magnitude[rows], self.n_inside[rows])
            ok = good
            self.stats["matched"] += int(good.sum())
        return feats, ok


class ReliabilityFunction:
    def __init__(self, base: LogisticHead, fly: LogisticHead | None = None, lookup: FlyLookup | None = None) -> None:
        self.base, self.fly, self.lookup = base, fly, lookup

    def __call__(self, pred_xywh: np.ndarray, dis_xyxy: np.ndarray | None, raw_index: int) -> np.ndarray:
        Xb = base_features(pred_xywh, dis_xyxy)
        p = self.base(Xb)
        if self.fly is not None and self.lookup is not None:
            Xf, ok = self.lookup.query(raw_index, pred_xywh)
            if ok.any():
                p[ok] = self.fly(np.concatenate([Xb[ok], Xf[ok]], axis=1))
        return p


def load_reliability_head(path: Path, fly_npz: Path | None = None, min_iou: float = 0.8) -> ReliabilityFunction:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    base = LogisticHead(spec["base"])
    fly = LogisticHead(spec["fly"]) if spec.get("fly") and fly_npz is not None else None
    lookup = FlyLookup(fly_npz, min_iou) if fly is not None else None
    return ReliabilityFunction(base, fly, lookup)
