"""Сравнить сохранение ID на переходах с окклюзией и без неё."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.occlusion_transition_eval import evaluate_result_directory  # noqa: E402


VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
)
TRACKERS = (
    "bytetrack",
    "bytetrack_dis_all_gated_a1",
    "bytetrack_dis_quarter_all_gated_a1",
    "bytetrack_dis_quarter_all_gated_a1_agree0.25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "video_rate" / "combined" / "results",
    )
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--trackers", nargs="+", default=TRACKERS)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_ids = args.video_id or list(VIDEO_IDS)
    truth_dir = PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val"
    results = {
        tracker: evaluate_result_directory(
            args.results_root / tracker,
            video_ids,
            truth_dir=truth_dir,
            iou_threshold=args.iou,
        )
        for tracker in args.trackers
    }
    output = args.output or args.results_root.parent / "occlusion_transitions.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {tracker: value["combined"] for tracker, value in results.items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"[OK] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
