"""Измерить масштаб FlyVis на синтетических переносах известной величины.

Это диагностическая калибровка, а не обучение на BDD100K. Она проверяет знак,
линейность и способ агрегации десяти внутренних шагов между наблюдениями 5 Гц.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.flyvis_motion import FlyVisMotionEstimator  # noqa: E402


OUTPUT = PROJECT_ROOT / "runs" / "flyvis" / "motion_calibration.json"
SHIFTS = ((5, 0), (10, 0), (20, 0), (-20, 0), (0, 5), (0, 10), (0, -20))
AGGREGATIONS = ("mean", "last", "sum")


def textured_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    rng = np.random.default_rng(20260912)
    gray = rng.normal(127, 45, size=(height, width)).clip(0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    for index in range(30):
        x = int(rng.integers(0, width - 80))
        y = int(rng.integers(0, height - 50))
        color = int(rng.integers(30, 225))
        cv2.rectangle(gray, (x, y), (x + 80, y + 50), color, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def shifted(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def fitted_gain(rows: list[dict]) -> tuple[float, float]:
    predicted = np.asarray([row["predicted_xy"] for row in rows], dtype=np.float64)
    target = np.asarray([row["true_xy"] for row in rows], dtype=np.float64)
    denominator = float(np.sum(predicted * predicted))
    gain = float(np.sum(predicted * target) / denominator) if denominator else 0.0
    rmse = float(np.sqrt(np.mean((predicted * gain - target) ** 2)))
    return gain, rmse


def main() -> int:
    base = textured_frame()
    estimator = FlyVisMotionEstimator(
        mode="full",
        source_fps=5.0,
        dt=0.02,
        flow_gain=1.0,
        temporal_resampling="hold",
        flow_aggregation="mean",
    )
    records = []
    for resampling in ("hold", "linear"):
        estimator.temporal_resampling = resampling
        for dx, dy in SHIFTS:
            estimator.reset()
            estimator.observe(base)
            estimator.observe(shifted(base, dx, dy))
            for aggregation in AGGREGATIONS:
                field = estimator.field_from_last(aggregation)
                predicted = np.median(field.vectors_xy, axis=0)
                records.append(
                    {
                        "temporal_resampling": resampling,
                        "aggregation": aggregation,
                        "true_xy": [dx, dy],
                        "predicted_xy": [float(predicted[0]), float(predicted[1])],
                    }
                )

    fits = []
    for resampling in ("hold", "linear"):
        for aggregation in AGGREGATIONS:
            subset = [
                row
                for row in records
                if row["temporal_resampling"] == resampling
                and row["aggregation"] == aggregation
            ]
            gain, rmse = fitted_gain(subset)
            direction_ok = all(
                np.dot(row["true_xy"], row["predicted_xy"]) > 0 for row in subset
            )
            fits.append(
                {
                    "temporal_resampling": resampling,
                    "aggregation": aggregation,
                    "fitted_gain": gain,
                    "rmse_pixels_after_gain": rmse,
                    "all_directions_correct": bool(direction_ok),
                }
            )
    result = {
        "scope": "synthetic translation scale diagnostic; not BDD100K fitting",
        "source_fps": 5.0,
        "dt": 0.02,
        "model": "flow/0000/000",
        "records": records,
        "fits": fits,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(fits, ensure_ascii=False, indent=2))
    print(f"[OK] отчёт: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

