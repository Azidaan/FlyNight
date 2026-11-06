"""Калибровать масштаб FlyVis на синтетическом движении заданной частоты.

Разметка BDD100K не используется. Текстурированное изображение движется с
известным постоянным смещением между соседними кадрами, поэтому коэффициент
перевода выхода декодера в пиксели оценивается до проверки трекера.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.flyvis_motion_calibration import shifted, textured_frame  # noqa: E402
from tracking.flyvis_motion import FlyVisMotionEstimator, TileLayout  # noqa: E402


VELOCITIES = ((1, 0), (2, 0), (4, 0), (-2, 0), (0, 1), (0, 2), (0, -2))
AGGREGATIONS = ("mean", "last", "sum")


def fit_gain(records: list[dict]) -> dict:
    predicted = np.asarray([row["predicted_xy"] for row in records], np.float64)
    target = np.asarray([row["target_xy"] for row in records], np.float64)
    denominator = float(np.sum(predicted * predicted))
    gain = float(np.sum(predicted * target) / denominator) if denominator else 0.0
    calibrated = predicted * gain
    endpoint_error = np.linalg.norm(calibrated - target, axis=1)
    dot = np.sum(calibrated * target, axis=1)
    return {
        "samples": len(records),
        "fitted_gain": gain,
        "mean_endpoint_error_pixels": float(endpoint_error.mean()),
        "median_endpoint_error_pixels": float(np.median(endpoint_error)),
        "p90_endpoint_error_pixels": float(np.quantile(endpoint_error, 0.90)),
        "direction_accuracy": float(np.mean(dot > 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--max-dt", type=float, default=0.02)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flyvis" / "video_rate_calibration.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= args.warmup_frames + 1:
        raise ValueError("После прогрева должны остаться минимум два кадра")
    base = textured_frame()
    records = []
    integration = None
    for mode in ("full", "tiled"):
        estimator = FlyVisMotionEstimator(
            mode=mode,
            layout=TileLayout(2, 3, 0.20),
            source_fps=args.source_fps,
            dt=args.max_dt,
            flow_gain=1.0,
            temporal_resampling="linear",
            flow_aggregation="mean",
        )
        integration = {
            "steps_per_video_frame": estimator.integration_steps,
            "integration_dt_seconds": estimator.dt,
            "requested_max_dt_seconds": estimator.requested_max_dt,
        }
        for resampling in ("hold", "linear"):
            estimator.temporal_resampling = resampling
            for dx, dy in VELOCITIES:
                estimator.reset()
                for frame_index in range(args.frames):
                    frame = shifted(base, dx * frame_index, dy * frame_index)
                    estimator.observe(frame)
                    if frame_index == 0:
                        continue
                    for aggregation in AGGREGATIONS:
                        field = estimator.field_from_last(aggregation)
                        predicted = np.median(field.vectors_xy, axis=0)
                        if frame_index >= args.warmup_frames:
                            records.append(
                                {
                                    "mode": mode,
                                    "temporal_resampling": resampling,
                                    "aggregation": aggregation,
                                    "frame_index": frame_index,
                                    "target_xy": [dx, dy],
                                    "predicted_xy": predicted.astype(float).tolist(),
                                }
                            )
                print(
                    f"[CAL] {mode}/{resampling}: скорость ({dx}, {dy})",
                    flush=True,
                )

    fits = []
    for mode in ("full", "tiled"):
        for resampling in ("hold", "linear"):
            for aggregation in AGGREGATIONS:
                selected = [
                    row
                    for row in records
                    if row["mode"] == mode
                    and row["temporal_resampling"] == resampling
                    and row["aggregation"] == aggregation
                ]
                fits.append(
                    {
                        "mode": mode,
                        "temporal_resampling": resampling,
                        "aggregation": aggregation,
                        **fit_gain(selected),
                    }
                )
    result = {
        "scope": "synthetic constant-translation calibration; BDD100K unused",
        "source_fps": args.source_fps,
        "frames_per_sequence": args.frames,
        "discarded_warmup_frames": args.warmup_frames,
        "velocities_pixels_per_video_frame": VELOCITIES,
        "integration": integration,
        "fits": fits,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(fits, ensure_ascii=False, indent=2))
    print(f"[OK] отчёт: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
