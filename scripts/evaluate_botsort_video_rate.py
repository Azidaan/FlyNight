"""Оценить BoT-SORT с компенсацией движения камеры на исходных видео."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    detections_for_frame,
    distribution,
    load_detection_cache,
    save_json,
    scalabel_frame,
)
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402


VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
    "b232c7c9-d251d9ee",
    "b1d22449-15fb948f",
    "b251064f-e7a165fd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate"
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument(
        "--gmc",
        choices=("sparseOptFlow", "orb", "sift", "ecc", "none"),
        default="sparseOptFlow",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def botsort_args(gmc_method: str):
    args = make_tracker_args()
    args.gmc_method = gmc_method
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.8
    args.with_reid = False
    args.model = "auto"
    return args


def main() -> int:
    from ultralytics.trackers.bot_sort import BOTSORT

    args = parse_args()
    video_ids = args.video_id or list(VIDEO_IDS)
    name = f"botsort_{args.gmc.lower()}_no_reid"
    result_dir = args.root / "combined" / "results" / name
    timings = []
    per_video_timing = {}
    for video_id in video_ids:
        experiment = args.root / video_id
        detections = load_detection_cache(experiment / "detections" / "raw_video.npz")
        alignment = json.loads(
            (experiment / "alignment.json").read_text(encoding="utf-8")
        )
        truth_path = (
            PROJECT_ROOT
            / "data"
            / "bdd100k"
            / "labels"
            / "box_track_20"
            / "val"
            / f"{video_id}.json"
        )
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        anchor_by_raw = {
            int(row["video_frame_index"]): index
            for index, row in enumerate(alignment["matches"])
        }
        capture = cv2.VideoCapture(
            str(PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val" / f"{video_id}.mov")
        )
        if not capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео {video_id}")
        tracker = BOTSORT(botsort_args(args.gmc))
        rows = []
        local_timings = []
        for frame_index in range(int(detections["frame_count"])):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{video_id}: декодирование остановилось на {frame_index}")
            xyxy, confidence, classes = detections_for_frame(detections, frame_index)
            started = time.perf_counter()
            output = tracker.update(_DetSlice(xyxy, confidence, classes), img=frame)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            local_timings.append(elapsed)
            if frame_index in anchor_by_raw:
                reference_index = anchor_by_raw[frame_index]
                rows.append(scalabel_frame(video_id, truth[reference_index], output))
        capture.release()
        save_json(result_dir / f"{video_id}.json", rows)
        per_video_timing[video_id] = distribution(local_timings)
        print(f"[BOTSORT] {video_id}", flush=True)
    metrics = evaluate_hota(result_dir, video_ids)
    output = {
        "video_ids": video_ids,
        "gmc_method": args.gmc,
        "with_reid": False,
        "timing_excluding_decode_detector": distribution(timings),
        "per_video_timing": per_video_timing,
        "metrics": metrics,
    }
    output_path = args.output or args.root / "combined" / f"{name}.json"
    save_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[OK] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
