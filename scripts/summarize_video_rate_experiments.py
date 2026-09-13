"""Агрегировать 30-герцовые прогоны и построить честные 5-герцовые контроли."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    DEFAULT_VIDEO_ID,
    detections_for_frame,
    load_detection_cache,
    save_json,
    scalabel_frame,
)
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402


DEFAULT_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)
DEFAULT_TRACKERS = (
    "bytetrack",
    "bytetrack_dis_all_gated_a1",
    "bytetrack_dis_half_all_gated_a1",
    "bytetrack_dis_quarter_all_gated_a1",
    "bytetrack_dis_quarter_all_gated_a1_agree0.25",
    "constant_velocity",
    "dis",
    "flyvis_full_calibrated",
    "flyvis_tiled_calibrated",
)

MOTION_BY_TRACKER = {
    "dis": "dis",
    "flyvis_full_calibrated": "flyvis_full",
    "flyvis_tiled_calibrated": "flyvis_tiled",
    "bytetrack_dis_all_gated_a1": "dis",
    "bytetrack_dis_half_all_gated_a1": "dis_half",
    "bytetrack_dis_quarter_all_gated_a1": "dis_quarter",
    "bytetrack_dis_quarter_all_gated_a1_agree0.25": "dis_quarter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate"
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--trackers", nargs="+", default=DEFAULT_TRACKERS)
    return parser.parse_args()


def copy_results(root: Path, combined: Path, video_ids: list[str], tracker: str) -> Path:
    target = combined / "results" / tracker
    target.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        source = root / video_id / "results" / tracker / f"{video_id}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / source.name)
    return target


def run_anchor_only_bytetrack(
    root: Path,
    combined: Path,
    video_ids: list[str],
    *,
    track_buffer: int,
    name: str,
) -> Path:
    from ultralytics.trackers.byte_tracker import BYTETracker

    target = combined / "results" / name
    target.mkdir(parents=True, exist_ok=True)
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
        tracker = BYTETracker(make_tracker_args(track_buffer=track_buffer))
        rows = []
        for reference_index, match in enumerate(alignment["matches"]):
            raw_index = int(match["video_frame_index"])
            xyxy, confidence, classes = detections_for_frame(detections, raw_index)
            output = tracker.update(_DetSlice(xyxy, confidence, classes))
            rows.append(scalabel_frame(video_id, truth[reference_index], output))
        save_json(target / f"{video_id}.json", rows)
    return target


def weighted_mean(summaries: dict, path: tuple[str, ...]) -> float:
    total, frames = 0.0, 0
    for summary in summaries.values():
        value = summary
        for key in path:
            value = value[key]
        count = int(summary["protocol"]["raw_frames"])
        total += float(value) * count
        frames += count
    return total / frames


def association_mean_ms(root: Path, summaries: dict, tracker: str) -> float:
    total, frames = 0.0, 0
    for video_id, summary in summaries.items():
        count = int(summary["protocol"]["raw_frames"])
        if tracker in summary["trackers"]:
            value = summary["trackers"][tracker]["timing"]["association"]["mean_ms"]
        else:
            injection = json.loads(
                (root / video_id / "bytetrack_motion_injection_gated.json").read_text(
                    encoding="utf-8"
                )
            )
            value = injection[tracker]["timing"]["mean_ms"]
        total += float(value) * count
        frames += count
    return total / frames


def motion_mean_ms(root: Path, summaries: dict, source: str) -> float:
    total, frames = 0.0, 0
    for video_id, summary in summaries.items():
        count = int(summary["protocol"]["raw_frames"])
        if source in summary["motion_sources"]:
            value = summary["motion_sources"][source]["observation"]["mean_ms"]
        else:
            metadata = json.loads(
                (root / video_id / "motion" / f"{source}.json").read_text(
                    encoding="utf-8"
                )
            )
            value = metadata["observation"]["mean_ms"]
        total += float(value) * count
        frames += count
    return total / frames


def main() -> int:
    args = parse_args()
    video_ids = args.video_id or list(DEFAULT_IDS)
    combined = args.root / "combined"
    summaries = {
        video_id: json.loads(
            (args.root / video_id / "summary.json").read_text(encoding="utf-8")
        )
        for video_id in video_ids
    }

    metrics = {}
    for tracker in args.trackers:
        result_dir = copy_results(args.root, combined, video_ids, tracker)
        metrics[tracker] = evaluate_hota(result_dir, video_ids)

    unchanged_5hz = run_anchor_only_bytetrack(
        args.root,
        combined,
        video_ids,
        track_buffer=30,
        name="bytetrack_anchor_only_buffer30",
    )
    matched_time_5hz = run_anchor_only_bytetrack(
        args.root,
        combined,
        video_ids,
        track_buffer=5,
        name="bytetrack_anchor_only_buffer5",
    )
    metrics["bytetrack_anchor_only_buffer30"] = evaluate_hota(
        unchanged_5hz, video_ids
    )
    metrics["bytetrack_anchor_only_buffer5"] = evaluate_hota(
        matched_time_5hz, video_ids
    )

    required_motion_sources = sorted(
        {MOTION_BY_TRACKER[name] for name in args.trackers if name in MOTION_BY_TRACKER}
    )
    timing = {
        "decode_mean_ms": weighted_mean(summaries, ("detector", "decode", "mean_ms")),
        "detector_mean_ms": weighted_mean(
            summaries, ("detector", "detector", "mean_ms")
        ),
        "motion_mean_ms": {
            source: motion_mean_ms(args.root, summaries, source)
            for source in required_motion_sources
        },
        "association_mean_ms": {
            tracker: association_mean_ms(args.root, summaries, tracker)
            for tracker in args.trackers
        },
    }
    decode = timing["decode_mean_ms"]
    detector = timing["detector_mean_ms"]
    pipeline_fps = {}
    for tracker in args.trackers:
        source = MOTION_BY_TRACKER.get(tracker)
        motion = timing["motion_mean_ms"].get(source, 0.0)
        total_ms = decode + detector + motion + timing["association_mean_ms"][tracker]
        pipeline_fps[tracker] = {
            "component_sum_ms": total_ms,
            "fps_from_component_sum": 1000.0 / total_ms,
        }
    timing["pipeline_estimate"] = pipeline_fps

    endpoint = {}
    for tracker in ("dis", "flyvis_full_calibrated", "flyvis_tiled_calibrated"):
        if tracker not in args.trackers or not all(
            tracker in row.get("motion_endpoint_audits", {})
            for row in summaries.values()
        ):
            continue
        samples = sum(
            row["motion_endpoint_audits"][tracker]["samples"]
            for row in summaries.values()
        )
        endpoint[tracker] = {
            "samples": samples,
            "mean_endpoint_error_pixels": sum(
                row["motion_endpoint_audits"][tracker]["mean_endpoint_error_pixels"]
                * row["motion_endpoint_audits"][tracker]["samples"]
                for row in summaries.values()
            )
            / samples,
        }

    output = {
        "video_ids": video_ids,
        "raw_frames": sum(row["protocol"]["raw_frames"] for row in summaries.values()),
        "labeled_frames": sum(
            row["protocol"]["labeled_frames"] for row in summaries.values()
        ),
        "metrics": metrics,
        "timing": timing,
        "motion_endpoint": endpoint,
        "frequency_control": (
            "30 Hz and anchor-only controls use detections from the exact same decoded "
            "raw frames; buffer30 is unchanged ByteTrack, buffer5 approximates the same "
            "one-second retention as buffer30 at 30 Hz"
        ),
    }
    save_json(combined / "summary.json", output)
    ranking = {
        name: {
            key: value
            for key, value in row.items()
            if key in {"HOTA", "DetA", "AssA", "IDF1", "IDSw", "MOTA"}
        }
        for name, row in metrics.items()
    }
    print(json.dumps(ranking, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(timing["pipeline_estimate"], ensure_ascii=False, indent=2))
    print(f"[OK] {combined / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
