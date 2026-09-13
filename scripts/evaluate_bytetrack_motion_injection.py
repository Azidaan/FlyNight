"""Проверить добавление внешнего движения в прогноз неизменённого ByteTrack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    DEFAULT_GAINS,
    DEFAULT_VIDEO_ID,
    detections_for_frame,
    distribution,
    load_detection_cache,
    load_motion_cache,
    save_json,
    scalabel_frame,
)
from tracking.flyvis_tracker import MotionField  # noqa: E402
from tracking.motion_augmented_bytetrack import MotionAugmentedByteTrack  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "video_rate" / DEFAULT_VIDEO_ID,
    )
    parser.add_argument(
        "--sources", nargs="+", default=("dis", "flyvis_full", "flyvis_tiled")
    )
    parser.add_argument("--blends", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument("--apply-to", nargs="+", default=("all", "lost"))
    parser.add_argument(
        "--without-confidence",
        action="store_true",
        help="не ослаблять поправку по пространственной согласованности поля",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detections = load_detection_cache(args.experiment_dir / "detections" / "raw_video.npz")
    alignment = json.loads(
        (args.experiment_dir / "alignment.json").read_text(encoding="utf-8")
    )
    matches = alignment["matches"]
    truth_path = (
        PROJECT_ROOT
        / "data"
        / "bdd100k"
        / "labels"
        / "box_track_20"
        / "val"
        / f"{args.video_id}.json"
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    anchor_by_raw = {
        int(row["video_frame_index"]): index for index, row in enumerate(matches)
    }
    summaries = {}
    for source in args.sources:
        gain = DEFAULT_GAINS.get(source, 1.0)
        points, vectors = load_motion_cache(
            args.experiment_dir / "motion" / f"{source}.npz", gain
        )
        for apply_to in args.apply_to:
            for blend in args.blends:
                confidence_tag = "fixed" if args.without_confidence else "gated"
                name = f"bytetrack_{source}_{apply_to}_{confidence_tag}_a{blend:g}"
                print(f"[RUN] {name}", flush=True)
                tracker = MotionAugmentedByteTrack(
                    make_tracker_args(),
                    blend=blend,
                    apply_to=apply_to,
                    use_confidence=not args.without_confidence,
                )
                rows, timings = [], []
                frame_count = int(detections["frame_count"])
                for frame_index in range(frame_count):
                    xyxy, confidence, classes = detections_for_frame(
                        detections, frame_index
                    )
                    tracker.set_motion_field(MotionField(points, vectors[frame_index]))
                    started = time.perf_counter()
                    output = tracker.update(_DetSlice(xyxy, confidence, classes))
                    timings.append(time.perf_counter() - started)
                    if frame_index in anchor_by_raw:
                        reference_index = anchor_by_raw[frame_index]
                        rows.append(
                            scalabel_frame(
                                args.video_id, truth[reference_index], output
                            )
                        )
                result_dir = args.experiment_dir / "results" / name
                save_json(result_dir / f"{args.video_id}.json", rows)
                metrics = evaluate_hota(result_dir, [args.video_id])
                weights = [row["weight"] for row in tracker.motion_audit]
                summaries[name] = {
                    "source": source,
                    "source_gain": gain,
                    "apply_to": apply_to,
                    "blend": blend,
                    "use_confidence": not args.without_confidence,
                    "motion_updates": len(weights),
                    "mean_effective_weight": float(np.mean(weights)) if weights else 0.0,
                    "timing": distribution(timings),
                    "metrics": metrics,
                }
    suffix = "fixed" if args.without_confidence else "gated"
    output = args.experiment_dir / f"bytetrack_motion_injection_{suffix}.json"
    save_json(output, summaries)
    ranking = sorted(
        (
            {
                "name": name,
                "HOTA": row["metrics"]["HOTA"],
                "AssA": row["metrics"]["AssA"],
                "IDF1": row["metrics"]["IDF1"],
                "IDSw": row["metrics"]["IDSw"],
                "mean_effective_weight": row["mean_effective_weight"],
            }
            for name, row in summaries.items()
        ),
        key=lambda row: row["HOTA"],
        reverse=True,
    )
    print(json.dumps(ranking, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
