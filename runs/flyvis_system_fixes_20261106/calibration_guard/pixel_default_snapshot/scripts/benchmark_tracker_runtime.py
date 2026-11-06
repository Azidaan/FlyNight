"""Измерить скорость ByteTrack и FlyVisTracker при одном запуске YOLO26.

YOLO26 вычисляется один раз на кадр. Одни и те же массивы обнаружений затем
передаются ByteTrack и FlyVisTracker. В отчёте разделены декодирование JPEG,
детектор, ассоциация ByteTrack и движение с ассоциацией FlyVis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from tracking.detection_cache import load_video_cache  # noqa: E402
from tracking.flyvis_motion import GEOMETRY_VERSION, GEOMETRY_VERSIONS, FlyVisMotionEstimator, TileLayout, resolve_flow_gain  # noqa: E402
from tracking.flyvis_tracker import FlyVisTracker, FlyVisTrackerConfig  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402


DEFAULT_VIDEOS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distribution(milliseconds: list[float]) -> dict:
    values = np.asarray(milliseconds, dtype=np.float64)
    if not len(values):
        return {"frames": 0}
    mean = float(values.mean())
    return {
        "frames": int(len(values)),
        "mean_ms": mean,
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "fps_from_total_time": float(1000.0 / mean),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "det_cache_ctl_960"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--geometry-version", choices=GEOMETRY_VERSIONS, default=GEOMETRY_VERSION)
    parser.add_argument(
        "--half",
        action="store_true",
        help="использовать FP16 для YOLO26 на CUDA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "runtime_benchmark" / "tracker_runtime.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Отчёт уже существует: {args.output}; укажите новый --output")
    resolve_flow_gain(None, args.geometry_version)  # reject uncalibrated mode before detector warmup
    videos = args.video_id or list(DEFAULT_VIDEOS)
    from ultralytics import YOLO
    from ultralytics.trackers.byte_tracker import BYTETracker

    model = YOLO(str(args.weights))
    image_root = DATA_ROOT / "images" / "track" / args.split
    first_cache = load_video_cache(args.cache_dir, args.split, videos[0])
    first_name = str(first_cache["frame_names"][0])
    first_frame = cv2.imread(str(image_root / videos[0] / first_name), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError("Не удалось прочитать первый кадр для прогрева")
    model.predict(
        first_frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        quantize=16 if args.half else None,
        verbose=False,
    )
    prediction_device = torch.device(model.predictor.device)
    synchronize(prediction_device)

    estimator = FlyVisMotionEstimator(
        mode="tiled",
        layout=TileLayout(rows=2, columns=3, overlap=0.20),
        source_fps=5.0,
        dt=0.02,
        geometry_version=args.geometry_version,
    )
    flyvis_config = FlyVisTrackerConfig(update_velocity_from_matches=True)
    flyvis_device = next(estimator.network.parameters()).device

    decode_ms: list[float] = []
    detector_ms: list[float] = []
    bytetrack_ms: list[float] = []
    flyvis_ms: list[float] = []
    detector_plus_bytetrack_ms: list[float] = []
    detector_plus_flyvis_ms: list[float] = []
    stream_bytetrack_ms: list[float] = []
    stream_flyvis_ms: list[float] = []
    per_video = []

    for video_number, video_id in enumerate(videos, 1):
        cache = load_video_cache(args.cache_dir, args.split, video_id)
        frame_names = [str(name) for name in cache["frame_names"]]
        if args.max_frames is not None:
            frame_names = frame_names[: args.max_frames]
        byte_tracker = BYTETracker(make_tracker_args())
        fly_tracker = FlyVisTracker(flyvis_config, estimator)
        fly_tracker.reset()
        start_count = len(detector_ms)

        for frame_name in frame_names:
            decode_started = time.perf_counter()
            frame = cv2.imread(str(image_root / video_id / frame_name), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Не удалось прочитать кадр {video_id}/{frame_name}")
            decoded_at = time.perf_counter()

            synchronize(prediction_device)
            detector_started = time.perf_counter()
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                quantize=16 if args.half else None,
                verbose=False,
            )[0]
            xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            class_id = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            synchronize(prediction_device)
            detected_at = time.perf_counter()

            byte_started = detected_at
            byte_tracker.update(_DetSlice(xyxy, confidence, class_id))
            byte_finished = time.perf_counter()

            synchronize(flyvis_device)
            fly_started = time.perf_counter()
            fly_tracker.update(xyxy, confidence, class_id, frame=frame)
            synchronize(flyvis_device)
            fly_finished = time.perf_counter()

            decode_value = (decoded_at - decode_started) * 1000.0
            detector_value = (detected_at - detector_started) * 1000.0
            byte_value = (byte_finished - byte_started) * 1000.0
            fly_value = (fly_finished - fly_started) * 1000.0
            decode_ms.append(decode_value)
            detector_ms.append(detector_value)
            bytetrack_ms.append(byte_value)
            flyvis_ms.append(fly_value)
            detector_plus_bytetrack_ms.append(detector_value + byte_value)
            detector_plus_flyvis_ms.append(detector_value + fly_value)
            stream_bytetrack_ms.append(decode_value + detector_value + byte_value)
            stream_flyvis_ms.append(decode_value + detector_value + fly_value)

        end_count = len(detector_ms)
        per_video.append(
            {
                "video_id": video_id,
                "frames": end_count - start_count,
                "yolo_bytetrack": distribution(
                    detector_plus_bytetrack_ms[start_count:end_count]
                ),
                "yolo_flyvis_tiled_fused": distribution(
                    detector_plus_flyvis_ms[start_count:end_count]
                ),
            }
        )
        print(
            f"[{video_number}/{len(videos)}] {video_id}: {end_count - start_count} кадров",
            flush=True,
        )

    result = {
        "scope": "same-frame wall-clock runtime; one YOLO26 inference shared by both trackers",
        "includes_model_initialization": False,
        "flyvis_stream_startup_fade_in_included": True,
        "input_frame_rate_hz": 5.0,
        "geometry_version": args.geometry_version,
        "videos": videos,
        "weights": str(args.weights.resolve()),
        "detector": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "quantize": 16 if args.half else None,
        },
        "hardware": {
            "cuda": prediction_device.type == "cuda" or flyvis_device.type == "cuda",
            "gpu": torch.cuda.get_device_name(prediction_device) if prediction_device.type == "cuda" else None,
            "detector_device": str(prediction_device),
            "flyvis_device": str(flyvis_device),
            "flyvis_gpu": torch.cuda.get_device_name(flyvis_device) if flyvis_device.type == "cuda" else None,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
        "components": {
            "jpeg_decode": distribution(decode_ms),
            "yolo26": distribution(detector_ms),
            "bytetrack_association": distribution(bytetrack_ms),
            "flyvis_tiled_fused_motion_and_association": distribution(flyvis_ms),
        },
        "pipelines_excluding_jpeg_decode": {
            "yolo26_bytetrack": distribution(detector_plus_bytetrack_ms),
            "yolo26_flyvis_tiled_fused": distribution(detector_plus_flyvis_ms),
        },
        "pipelines_including_jpeg_decode": {
            "yolo26_bytetrack": distribution(stream_bytetrack_ms),
            "yolo26_flyvis_tiled_fused": distribution(stream_flyvis_ms),
        },
        "per_video": per_video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] отчёт: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
