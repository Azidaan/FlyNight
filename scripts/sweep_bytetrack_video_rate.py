"""Настроить ByteTrack на всех кадрах исходных 30-герцовых видео."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

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


TRAIN_VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate"
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--high", nargs="+", type=float, default=(0.20, 0.25))
    parser.add_argument("--low", nargs="+", type=float, default=(0.03, 0.05, 0.10))
    parser.add_argument("--buffer", nargs="+", type=int, default=(30, 60))
    parser.add_argument("--match", nargs="+", type=float, default=(0.80,))
    parser.add_argument(
        "--tracker-class",
        choices=("byte", "xywh", "xywh_iou_second"),
        default="byte",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "video_rate" / "bytetrack_sweep_train4.json",
    )
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def config_name(high: float, low: float, buffer: int, match: float) -> str:
    return f"h{token(high)}_l{token(low)}_b{buffer}_m{token(match)}"


def run_configuration(
    root: Path,
    video_ids: list[str],
    result_dir: Path,
    *,
    high: float,
    low: float,
    buffer: int,
    match: float,
    tracker_class: str = "byte",
) -> dict:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from tracking.xywh_bytetrack import XYWHByteTrack, XYWHIoUSecondByteTrack

    tracker_types = {
        "byte": BYTETracker,
        "xywh": XYWHByteTrack,
        "xywh_iou_second": XYWHIoUSecondByteTrack,
    }
    tracker_type = tracker_types[tracker_class]

    timings = []
    for video_id in video_ids:
        experiment = root / video_id
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
        tracker = tracker_type(
            make_tracker_args(
                track_high_thresh=high,
                track_low_thresh=low,
                track_buffer=buffer,
                match_thresh=match,
            )
        )
        rows = []
        for frame_index in range(int(detections["frame_count"])):
            xyxy, confidence, classes = detections_for_frame(detections, frame_index)
            started = time.perf_counter()
            output = tracker.update(_DetSlice(xyxy, confidence, classes))
            timings.append(time.perf_counter() - started)
            if frame_index in anchor_by_raw:
                reference_index = anchor_by_raw[frame_index]
                rows.append(scalabel_frame(video_id, truth[reference_index], output))
        save_json(result_dir / f"{video_id}.json", rows)
    return {
        "timing": distribution(timings),
        "metrics": evaluate_hota(result_dir, video_ids),
    }


def main() -> int:
    args = parse_args()
    video_ids = args.video_id or list(TRAIN_VIDEO_IDS)
    configurations = [
        values
        for values in itertools.product(args.high, args.low, args.buffer, args.match)
        if values[1] < values[0]
    ]
    results = {}
    for index, (high, low, buffer, match) in enumerate(configurations, 1):
        name = config_name(high, low, buffer, match)
        print(f"[SWEEP] {index}/{len(configurations)} {name}", flush=True)
        result_dir = args.output.parent / args.output.stem / name
        row = run_configuration(
            args.root,
            video_ids,
            result_dir,
            high=high,
            low=low,
            buffer=buffer,
            match=match,
            tracker_class=args.tracker_class,
        )
        results[name] = {
            "parameters": {
                "track_high_thresh": high,
                "track_low_thresh": low,
                "track_buffer": buffer,
                "match_thresh": match,
            },
            **row,
        }
    output = {
        "video_ids": video_ids,
        "selection_role": "tuning" if video_ids == list(TRAIN_VIDEO_IDS) else "validation",
        "tracker_class": args.tracker_class,
        "results": results,
    }
    save_json(args.output, output)
    ranking = sorted(
        (
            {
                "name": name,
                **row["parameters"],
                "HOTA": row["metrics"]["HOTA"],
                "AssA": row["metrics"]["AssA"],
                "IDF1": row["metrics"]["IDF1"],
                "IDSw": row["metrics"]["IDSw"],
            }
            for name, row in results.items()
        ),
        key=lambda row: row["HOTA"],
        reverse=True,
    )
    print(json.dumps(ranking, ensure_ascii=False, indent=2))
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
