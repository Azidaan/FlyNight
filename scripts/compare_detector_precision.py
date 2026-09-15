"""Сравнить обнаружения YOLO26 FP16 с исходным кэшем FP32."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402
from tracking.flyvis_tracker import box_iou  # noqa: E402


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
    parser.add_argument("--video-id", default="b1cebfb7-284f5117")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "runtime_benchmark" / "fp16_detection_match.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    cache = load_video_cache(args.cache_dir, args.split, args.video_id)
    cached_frames = list(iter_frames(cache))[: args.max_frames]
    image_root = DATA_ROOT / "images" / "track" / args.split / args.video_id
    model = YOLO(str(args.weights))
    first = cv2.imread(str(image_root / cached_frames[0][1]), cv2.IMREAD_COLOR)
    model.predict(
        first,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0,
        quantize=16,
        verbose=False,
    )

    cached_total = 0
    fp16_total = 0
    matched_50 = 0
    matched_99 = 0
    matched_999 = 0
    equal_count_frames = 0
    matched_ious = []
    coordinate_differences = []
    confidence_differences = []

    for _, frame_name, cached_xyxy, cached_conf, cached_cls in cached_frames:
        frame = cv2.imread(str(image_root / frame_name), cv2.IMREAD_COLOR)
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=0,
            quantize=16,
            verbose=False,
        )[0]
        fp16_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        fp16_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        fp16_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        cached_total += len(cached_xyxy)
        fp16_total += len(fp16_xyxy)
        equal_count_frames += int(len(cached_xyxy) == len(fp16_xyxy))
        if not len(cached_xyxy) or not len(fp16_xyxy):
            continue
        ious = box_iou(cached_xyxy, fp16_xyxy)
        same_class = cached_cls[:, None] == fp16_cls[None, :]
        cost = np.where(same_class, 1.0 - ious, 1e6)
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if not same_class[row, column] or ious[row, column] < 0.5:
                continue
            value = float(ious[row, column])
            matched_ious.append(value)
            matched_50 += 1
            matched_99 += int(value >= 0.99)
            matched_999 += int(value >= 0.999)
            coordinate_differences.extend(
                np.abs(cached_xyxy[row] - fp16_xyxy[column]).astype(float).tolist()
            )
            confidence_differences.append(
                abs(float(cached_conf[row]) - float(fp16_conf[column]))
            )

    result = {
        "scope": "FP16 live detections against the existing FP32 detection cache",
        "video_id": args.video_id,
        "frames": len(cached_frames),
        "cached_detections": cached_total,
        "fp16_detections": fp16_total,
        "frames_with_equal_detection_count": equal_count_frames,
        "matched_same_class_iou_50": matched_50,
        "matched_same_class_iou_99": matched_99,
        "matched_same_class_iou_999": matched_999,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "coordinate_abs_difference_mean_pixels": (
            float(np.mean(coordinate_differences)) if coordinate_differences else None
        ),
        "coordinate_abs_difference_max_pixels": (
            float(np.max(coordinate_differences)) if coordinate_differences else None
        ),
        "confidence_abs_difference_mean": (
            float(np.mean(confidence_differences)) if confidence_differences else None
        ),
        "confidence_abs_difference_max": (
            float(np.max(confidence_differences)) if confidence_differences else None
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        },
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
