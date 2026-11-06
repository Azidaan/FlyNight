"""Сравнить поля движения с реальным смещением размеченных объектов.

Разметка используется только для диагностики после вычисления поля и никогда
не подаётся трекеру. Подобранный здесь коэффициент масштаба также только
показывает систематическую ошибку и не применяется в оценённых прогонах.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from evaluation.tracker_lifecycle import read_truth  # noqa: E402
from tracking.classical_motion import OpenCVMotionEstimator  # noqa: E402
from tracking.flyvis_motion import (  # noqa: E402
    GEOMETRY_VERSION,
    GEOMETRY_VERSIONS,
    SYNTHETIC_FLOW_GAIN,
    resolve_flow_gain,
    FlyVisMotionEstimator,
    TileLayout,
)


DEFAULT_VIDEOS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)


def center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) * 0.5


def summarize(records: list[dict], mode: str, video_id: str | None) -> dict:
    selected = [
        row
        for row in records
        if row["mode"] == mode and (video_id is None or row["video_id"] == video_id)
    ]
    if not selected:
        return {"mode": mode, "video_id": video_id or "COMBINED", "samples": 0}
    predicted = np.asarray([row["predicted_xy"] for row in selected], np.float64)
    target = np.asarray([row["target_xy"] for row in selected], np.float64)
    error = np.linalg.norm(predicted - target, axis=1)
    target_magnitude = np.linalg.norm(target, axis=1)
    predicted_magnitude = np.linalg.norm(predicted, axis=1)
    normalizer = np.asarray([row["box_scale"] for row in selected], np.float64)
    moving = target_magnitude >= 1.0
    dot = np.sum(predicted * target, axis=1)
    cosine = dot / np.maximum(predicted_magnitude * target_magnitude, 1e-12)
    denominator = float(np.sum(predicted * predicted))
    gain = float(np.sum(predicted * target) / denominator) if denominator else 0.0
    fitted_error = np.linalg.norm(predicted * gain - target, axis=1)
    confidence = np.asarray([row["confidence"] for row in selected], np.float64)
    if np.ptp(confidence) > 0.0:
        correlation = spearmanr(confidence, -error).statistic
    else:
        correlation = np.nan
    return {
        "mode": mode,
        "video_id": video_id or "COMBINED",
        "samples": len(selected),
        "mean_epe_pixels": float(error.mean()),
        "median_epe_pixels": float(np.median(error)),
        "mean_normalized_epe": float(np.mean(error / normalizer)),
        "median_target_magnitude": float(np.median(target_magnitude)),
        "median_predicted_magnitude": float(np.median(predicted_magnitude)),
        "p95_predicted_magnitude": float(np.quantile(predicted_magnitude, 0.95)),
        "direction_accuracy_moving": float(np.mean(dot[moving] > 0)) if moving.any() else None,
        "median_cosine_moving": float(np.median(cosine[moving])) if moving.any() else None,
        "diagnostic_fitted_gain": gain,
        "mean_epe_after_diagnostic_gain": float(fitted_error.mean()),
        "confidence_error_spearman": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "median_confidence": float(np.median(confidence)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--temporal-resampling", choices=("hold", "linear"), default="hold"
    )
    parser.add_argument(
        "--flow-aggregation", choices=("mean", "last", "sum"), default="mean"
    )
    parser.add_argument("--flow-gain", type=float, default=None)
    parser.add_argument("--geometry-version", choices=GEOMETRY_VERSIONS, default=GEOMETRY_VERSION)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flyvis_tracker_motion_gate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.flow_gain = resolve_flow_gain(args.flow_gain, args.geometry_version)
    videos = args.video_id or list(DEFAULT_VIDEOS)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("motion_source_metrics*")):
        raise FileExistsError(f"Результаты уже существуют: {args.output}; укажите новый --output")
    estimators = {
        "dis": OpenCVMotionEstimator("dis", sample_step=8, geometry_version=args.geometry_version),
        "flyvis_full": FlyVisMotionEstimator(
            mode="full",
            temporal_resampling=args.temporal_resampling,
            flow_aggregation=args.flow_aggregation,
            flow_gain=args.flow_gain,
            geometry_version=args.geometry_version,
        ),
        "flyvis_tiled": FlyVisMotionEstimator(
            mode="tiled",
            layout=TileLayout(rows=2, columns=3, overlap=0.2),
            temporal_resampling=args.temporal_resampling,
            flow_aggregation=args.flow_aggregation,
            flow_gain=args.flow_gain,
            geometry_version=args.geometry_version,
        ),
    }
    records = []
    for video_number, video_id in enumerate(videos, 1):
        truth_path = (
            DATA_ROOT / "labels" / "box_track_20" / args.split / f"{video_id}.json"
        )
        truth, _, _ = read_truth(truth_path)
        frame_paths = sorted(
            (DATA_ROOT / "images" / "track" / args.split / video_id).glob("*.jpg")
        )
        if args.max_frames is not None:
            frame_paths = frame_paths[: args.max_frames]
        for estimator in estimators.values():
            estimator.reset()
        previous_truth = None
        previous_velocity: dict[str, np.ndarray] = {}

        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Не удалось прочитать кадр: {frame_path}")
            fields = {mode: estimator.observe(frame) for mode, estimator in estimators.items()}
            current_truth = truth[frame_index]
            if previous_truth is not None:
                current_index = {object_id: i for i, object_id in enumerate(current_truth.ids)}
                # Constant-velocity history is valid only for consecutive visible
                # observations.  Keeping stale entries across an annotation gap
                # would silently turn this into an arbitrary long-horizon prior.
                next_velocity: dict[str, np.ndarray] = {}
                for previous_index, object_id in enumerate(previous_truth.ids):
                    if object_id not in current_index:
                        continue
                    current_object_index = current_index[object_id]
                    if previous_truth.classes[previous_index] != current_truth.classes[current_object_index]:
                        continue
                    previous_box = previous_truth.boxes[previous_index]
                    current_box = current_truth.boxes[current_object_index]
                    target = center(current_box) - center(previous_box)
                    scale = float(np.sqrt(np.prod(previous_box[2:] - previous_box[:2])))
                    common = {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "class_id": int(previous_truth.classes[previous_index]),
                        "target_xy": target.astype(float).tolist(),
                        "box_scale": max(scale, 1.0),
                    }
                    records.append(
                        {
                            **common,
                            "mode": "zero",
                            "predicted_xy": [0.0, 0.0],
                            "confidence": 0.0,
                        }
                    )
                    if object_id in previous_velocity:
                        records.append(
                            {
                                **common,
                                "mode": "constant_velocity",
                                "predicted_xy": previous_velocity[object_id].astype(float).tolist(),
                                "confidence": 1.0,
                            }
                        )
                    for mode, field in fields.items():
                        estimate = field.estimate_box(previous_box)
                        records.append(
                            {
                                **common,
                                "mode": mode,
                                "predicted_xy": estimate.vector_xy.astype(float).tolist(),
                                "confidence": estimate.confidence,
                            }
                        )
                        if object_id in previous_velocity:
                            alpha = float(np.clip(estimate.confidence, 0.0, 1.0))
                            fused = (
                                alpha * estimate.vector_xy
                                + (1.0 - alpha) * previous_velocity[object_id]
                            )
                            records.append(
                                {
                                    **common,
                                    "mode": f"{mode}_fused_oracle_history",
                                    "predicted_xy": fused.astype(float).tolist(),
                                    "confidence": alpha,
                                }
                            )
                    next_velocity[object_id] = target
                previous_velocity = next_velocity
            previous_truth = current_truth
        print(f"[{video_number}/{len(videos)}] {video_id}: {len(frame_paths)} кадров", flush=True)

    modes = sorted({row["mode"] for row in records})
    summaries = [summarize(records, mode, None) for mode in modes]
    per_video = [summarize(records, mode, video_id) for video_id in videos for mode in modes]
    result = {
        "scope": "motion-source diagnostic against GT displacement; GT not used by trackers",
        "protocol": {
            "temporal_resampling": args.temporal_resampling,
            "flow_aggregation": args.flow_aggregation,
            "flow_gain": args.flow_gain,
            "geometry_version": args.geometry_version,
            "flow_gain_status": ("explicit_unvalidated_resize_diagnostic" if args.geometry_version == "resize_v2"
                                 else "legacy_empirical_vector_convention"),
        },
        "videos": videos,
        "modes": modes,
        "summaries": summaries,
        "per_video": per_video,
    }
    (args.output / "motion_source_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "motion_source_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output / "motion_source_metrics_per_video.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0]))
        writer.writeheader()
        writer.writerows(per_video)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
