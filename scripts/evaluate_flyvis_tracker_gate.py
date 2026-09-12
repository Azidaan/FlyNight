"""Собрать официальные метрики одновидеофайлового гейта FlyVisTracker.

Сценарий не усредняет готовые проценты вручную: каждый каталог оценивается
существующим адаптером TrackEval. Различия CLR_TP/FP/FN выводятся явно, чтобы
не приписывать изменившуюся полноту обнаружений качеству ассоциации.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402


METRICS = (
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDSw",
    "CLR_TP",
    "CLR_FP",
    "CLR_FN",
    "Frag",
)


def quiet_hota(result_dir: Path, video_ids: list[str]) -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        return evaluate_hota(result_dir, video_ids)


def output_signature(result_dir: Path, video_ids: list[str]) -> tuple:
    """Сравнить фактические рамки/классы/оценки, игнорируя track ID."""
    signature = []
    for video_id in video_ids:
        frames = json.loads((result_dir / f"{video_id}.json").read_text(encoding="utf-8"))
        for frame in frames:
            labels = []
            for label in frame.get("labels") or []:
                box = label["box2d"]
                labels.append(
                    (
                        label["category"],
                        float(label.get("score", 1.0)),
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                    )
                )
            signature.append((video_id, int(frame["frameIndex"]), tuple(sorted(labels))))
    return tuple(signature)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_tracker_gate"
    )
    parser.add_argument("--per-video", action="store_true")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=(
            "none",
            "constant_velocity",
            "farneback",
            "dis",
            "full",
            "tiled",
            "full_fused",
            "tiled_fused",
            "bytetrack",
        ),
    )
    args = parser.parse_args()
    selected = args.video_id or ["b1cebfb7-284f5117"]

    rows = []
    for mode in args.modes:
        result_dir = args.root / mode
        metrics = quiet_hota(result_dir, selected)
        timings = []
        for video_id in selected:
            timing_path = result_dir / f"{video_id}.timing.json"
            if timing_path.exists():
                timings.append(json.loads(timing_path.read_text(encoding="utf-8")))
        elapsed = sum(timing.get("elapsed_seconds", 0.0) for timing in timings)
        frames = sum(timing.get("frames", 0) for timing in timings)
        row = {
            "mode": mode,
            "videos": len(selected),
            **{metric: metrics.get(metric) for metric in METRICS},
            "frames_per_second_excluding_model_init": (
                frames / elapsed if elapsed else None
            ),
            "created_tracks": (
                sum(timing.get("created_tracks", 0) for timing in timings)
                if timings
                else None
            ),
        }
        rows.append(row)

    standalone_modes = [mode for mode in args.modes if mode != "bytetrack"]
    signatures = [output_signature(args.root / mode, selected) for mode in standalone_modes]
    same_observations = all(signature == signatures[0] for signature in signatures[1:])
    per_video_rows = []
    if args.per_video:
        for video_id in selected:
            for mode in args.modes:
                metrics = quiet_hota(args.root / mode, [video_id])
                per_video_rows.append(
                    {
                        "mode": mode,
                        "video_id": video_id,
                        **{metric: metrics.get(metric) for metric in METRICS},
                    }
                )
    result = {
        "scope": "implementation gate; no parameter tuning",
        "video_ids": selected,
        "standalone_outputs_same_boxes_classes_scores": same_observations,
        "warning": (
            "TrackEval TP/FP/FN may differ despite identical boxes because matching uses identity history; "
            "ByteTrack has different output detections"
        ),
        "rows": rows,
        "per_video_rows": per_video_rows,
    }
    json_path = args.root / "gate_metrics.json"
    csv_path = args.root / "gate_metrics.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if per_video_rows:
        per_video_path = args.root / "gate_metrics_per_video.csv"
        with per_video_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0]))
            writer.writeheader()
            writer.writerows(per_video_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
