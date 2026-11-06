"""Запустить самостоятельный FlyVisTracker на кэше обнаружений YOLO26.

Сценарий читает неизменные рамки из ``tracking.detection_cache``, но также
читает исходные кадры, необходимые FlyVis. Результат каждого видео сохраняется
в Scalabel JSON и подходит существующим оценщикам BDD100K.

Пример короткой проверки:
    .venv/Scripts/python.exe scripts/run_flyvis_tracker.py --mode full \
        --video-id b1cebfb7-284f5117 --max-frames 20 \
        --output runs/flyvis_tracker_smoke/full --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES, DATA_ROOT  # noqa: E402
from evaluation.tracker_lifecycle import LifecycleAudit, read_truth  # noqa: E402
from tracking.classical_motion import OpenCVMotionEstimator  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.flyvis_motion import (  # noqa: E402
    GEOMETRY_VERSION,
    GEOMETRY_VERSIONS,
    SYNTHETIC_FLOW_GAIN,
    FlyVisMotionEstimator,
    TileLayout,
)
from tracking.flyvis_tracker import (  # noqa: E402
    FlyVisTracker,
    FlyVisTrackerConfig,
    MotionField,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "det_cache_ctl_960"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "none",
            "constant_velocity",
            "farneback",
            "dis",
            "full",
            "tiled",
            "full_fused",
            "tiled_fused",
        ),
        default="full",
    )
    parser.add_argument("--classical-scale", type=float, default=1.0)
    parser.add_argument("--classical-sample-step", type=int, default=8)
    parser.add_argument("--tile-rows", type=int, default=2)
    parser.add_argument("--tile-columns", type=int, default=3)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--source-fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--flow-gain", type=float, default=SYNTHETIC_FLOW_GAIN)
    parser.add_argument("--geometry-version", choices=GEOMETRY_VERSIONS, default=GEOMETRY_VERSION)
    parser.add_argument(
        "--temporal-resampling", choices=("hold", "linear"), default="hold"
    )
    parser.add_argument(
        "--flow-aggregation", choices=("mean", "last", "sum"), default="mean"
    )
    parser.add_argument(
        "--keep-decoder-bias",
        action="store_true",
        help="не вычитать статический ответ декодера после инициализации",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--new-track-threshold", type=float, default=0.25)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--iou-weight", type=float, default=0.35)
    parser.add_argument("--motion-weight", type=float, default=0.55)
    parser.add_argument("--size-weight", type=float, default=0.10)
    parser.add_argument("--max-motion-distance", type=float, default=2.5)
    parser.add_argument("--max-assignment-cost", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_tracker"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def video_ids(args: argparse.Namespace) -> list[str]:
    cache_split = args.cache_dir / args.split
    if not cache_split.is_dir():
        raise FileNotFoundError(f"Каталог кэша не найден: {cache_split}")
    available = sorted(path.stem for path in cache_split.glob("*.npz"))
    selected = args.video_id or available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"В кэше нет видео: {', '.join(missing)}")
    if args.limit_videos is not None:
        selected = selected[: args.limit_videos]
    if not selected:
        raise RuntimeError("Не выбрано ни одного видео")
    return selected


def tracker_config(args: argparse.Namespace) -> FlyVisTrackerConfig:
    return FlyVisTrackerConfig(
        detection_threshold=args.detection_threshold,
        new_track_threshold=args.new_track_threshold,
        min_hits=args.min_hits,
        max_age=args.max_age,
        iou_weight=args.iou_weight,
        motion_weight=args.motion_weight,
        size_weight=args.size_weight,
        max_motion_distance=args.max_motion_distance,
        max_assignment_cost=args.max_assignment_cost,
        update_velocity_from_matches=args.mode
        in {"constant_velocity", "full_fused", "tiled_fused"},
    )


def scalabel_frame(video_id: str, frame_index: int, name: str, output) -> dict:
    labels = []
    for row in output:
        x1, y1, x2, y2, track_id, score, class_id = row[:7]
        labels.append(
            {
                "id": str(int(track_id)),
                "category": CLASS_NAMES[int(class_id)],
                "score": float(score),
                "box2d": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            }
        )
    return {
        "videoName": video_id,
        "name": name,
        "frameIndex": frame_index,
        "labels": labels,
    }


def motion_summary(history: list[dict]) -> dict:
    if not history:
        return {"samples": 0}
    magnitude = np.hypot(
        [row["dx"] for row in history], [row["dy"] for row in history]
    )
    confidence = np.asarray([row["confidence"] for row in history], dtype=np.float64)
    return {
        "samples": len(history),
        "magnitude_p50_pixels": float(np.quantile(magnitude, 0.50)),
        "magnitude_p90_pixels": float(np.quantile(magnitude, 0.90)),
        "magnitude_p99_pixels": float(np.quantile(magnitude, 0.99)),
        "magnitude_max_pixels": float(np.max(magnitude)),
        "confidence_p50": float(np.quantile(confidence, 0.50)),
        "confidence_p10": float(np.quantile(confidence, 0.10)),
    }


def diagnostic_lifecycle(video_id: str, split: str, rows: list[dict]) -> dict:
    """Диагностика IoU=0.5 на обработанном префиксе; это не HOTA."""
    truth_path = DATA_ROOT / "labels" / "box_track_20" / split / f"{video_id}.json"
    truth, _, _ = read_truth(truth_path)
    frame_indices = [int(frame["frameIndex"]) for frame in rows]
    first = {}
    for frame_index in frame_indices:
        for object_id in truth[frame_index].ids:
            first.setdefault(object_id, frame_index)
    audit = LifecycleAudit(first)
    class_ids = {name: index for index, name in CLASS_NAMES.items()}
    for frame in rows:
        output_rows = []
        for label in frame["labels"]:
            box = label["box2d"]
            output_rows.append(
                [
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                    int(label["id"]),
                    float(label["score"]),
                    class_ids[label["category"]],
                    -1,
                ]
            )
        audit.observe(
            int(frame["frameIndex"]),
            np.asarray(output_rows, dtype=np.float32).reshape(-1, 8),
            truth[int(frame["frameIndex"])],
        )
    summary = audit.summary()
    summary["scope"] = "prefix diagnostic at IoU>=0.5; not official HOTA"
    return summary


def validate_output_protocol(output: Path, metadata: dict) -> None:
    """Не смешивать видео разных протоколов, включая запуск с --force."""
    metadata_path = output / "_metadata.json"
    if not metadata_path.exists():
        if any(output.glob("*.json")):
            raise RuntimeError(f"{output} содержит результаты без протокола; укажите новый --output")
        return
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    if previous.get("mode") in {"full", "tiled", "full_fused", "tiled_fused", "farneback", "dis"}:
        # До resize_v2 версия не записывалась. Совместимость требует явного
        # --geometry-version legacy_v1, а не присвоения старым полям новой версии.
        previous.setdefault("geometry_version", "legacy_v1")
    comparable_keys = (set(metadata) | set(previous)) - {"video_ids"}
    if any(previous.get(key) != metadata.get(key) for key in comparable_keys):
        raise RuntimeError(
            f"{output} уже содержит другой протокол; укажите новый --output "
            "(для старой геометрии: --geometry-version legacy_v1). "
            "--force повторяет только тот же протокол"
        )


def main() -> int:
    args = parse_args()
    selected = video_ids(args)
    config = tracker_config(args)
    layout = TileLayout(args.tile_rows, args.tile_columns, args.tile_overlap)
    args.output.mkdir(parents=True, exist_ok=True)

    cache_meta_path = args.cache_dir / "meta.json"
    cache_meta = (
        json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if cache_meta_path.exists()
        else None
    )
    metadata = {
        "scope": "standalone FlyVisTracker; ByteTrack is not used",
        "partial_run": args.max_frames is not None,
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_metadata": cache_meta,
        "split": args.split,
        "video_ids": selected,
        "model": args.model,
        "mode": args.mode,
        "classical_scale": args.classical_scale,
        "classical_sample_step": args.classical_sample_step,
        "layout": asdict(layout),
        "source_fps": args.source_fps,
        "dt": args.dt,
        "flow_gain": args.flow_gain,
        "temporal_resampling": args.temporal_resampling,
        "flow_aggregation": args.flow_aggregation,
        "subtract_initial_baseline": not args.keep_decoder_bias,
        "tracker_config": asdict(config),
    }
    if args.mode in {"full", "tiled", "full_fused", "tiled_fused", "farneback", "dis"}:
        metadata["geometry_version"] = args.geometry_version
    metadata_path = args.output / "_metadata.json"
    validate_output_protocol(args.output, metadata)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    estimator = None
    if args.mode in {"farneback", "dis"}:
        estimator = OpenCVMotionEstimator(
            args.mode,
            processing_scale=args.classical_scale,
            sample_step=args.classical_sample_step,
            geometry_version=args.geometry_version,
        )
    elif args.mode in {"full", "tiled", "full_fused", "tiled_fused"}:
        estimator = FlyVisMotionEstimator(
            model=args.model,
            mode="tiled" if args.mode.startswith("tiled") else "full",
            layout=layout,
            source_fps=args.source_fps,
            dt=args.dt,
            flow_gain=args.flow_gain,
            temporal_resampling=args.temporal_resampling,
            flow_aggregation=args.flow_aggregation,
            subtract_initial_baseline=not args.keep_decoder_bias,
            geometry_version=args.geometry_version,
        )
    tracker = FlyVisTracker(config, estimator)
    image_root = DATA_ROOT / "images" / "track" / args.split

    for video_number, video_id in enumerate(selected, 1):
        output_path = args.output / f"{video_id}.json"
        timing_path = args.output / f"{video_id}.timing.json"
        if output_path.exists() and not args.force:
            print(f"[SKIP] {video_id}: результат уже существует", flush=True)
            continue
        tracker.reset()
        cache = load_video_cache(args.cache_dir, args.split, video_id)
        rows = []
        stats_before = dict(estimator.stats) if estimator is not None else None
        started = time.perf_counter()
        for frame_index, frame_name, xyxy, confidence, class_id in iter_frames(cache):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if estimator is None:
                output = tracker.update(
                    xyxy, confidence, class_id, motion_field=MotionField.empty()
                )
            else:
                frame_path = image_root / video_id / frame_name
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Не удалось прочитать кадр: {frame_path}")
                output = tracker.update(xyxy, confidence, class_id, frame=frame)
            rows.append(scalabel_frame(video_id, frame_index, frame_name, output))

        elapsed = time.perf_counter() - started
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        stats_delta = None
        if estimator is not None and stats_before is not None:
            stats_delta = {
                key: estimator.stats[key] - stats_before[key] for key in estimator.stats
            }
        summary = {
            "video_id": video_id,
            "frames": len(rows),
            "partial_run": args.max_frames is not None,
            "elapsed_seconds": elapsed,
            "frames_per_second": len(rows) / elapsed if elapsed else None,
            "created_tracks": tracker.next_track_id - 1,
            "visible_outputs": sum(len(frame["labels"]) for frame in rows),
            "active_tracks_at_end": len(tracker.active_tracks),
            "motion_estimator": stats_delta,
            "track_motion": motion_summary(tracker.motion_history),
            "diagnostic_lifecycle": diagnostic_lifecycle(
                video_id, args.split, rows
            ),
            "lifecycle_events": {
                name: sum(event["event"] == name for event in tracker.events)
                for name in sorted({event["event"] for event in tracker.events})
            },
        }
        timing_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{video_number}/{len(selected)}] {video_id}: {len(rows)} кадров, "
            f"{summary['frames_per_second']:.2f} кадр/с, "
            f"{summary['created_tracks']} ID",
            flush=True,
        )
    print(f"[OK] результаты: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
