"""Ранжировать BDD100K MOT-видео по измеряемым сложным условиям."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "video_rate" / "adverse_case_ranking.json",
    )
    return parser.parse_args()


def sampled(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[index] for index in np.unique(indices)]


def visual_statistics(paths: list[Path]) -> dict:
    means, dark, highlights, contrast, edges = [], [], [], [], []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Не удалось прочитать {path}")
        means.append(float(frame.mean()))
        dark.append(float(np.mean(frame <= 24)))
        highlights.append(float(np.mean(frame >= 248)))
        contrast.append(float(frame.std()))
        edges.append(float(np.mean(cv2.Canny(frame, 100, 200) > 0)))
    return {
        "sampled_frames": len(paths),
        "mean_luma": float(np.mean(means)),
        "dark_fraction": float(np.mean(dark)),
        "highlight_fraction": float(np.mean(highlights)),
        "contrast_std": float(np.mean(contrast)),
        "edge_fraction": float(np.mean(edges)),
    }


def annotation_statistics(frames: list[dict]) -> dict:
    labels = [label for frame in frames for label in (frame.get("labels") or [])]
    occluded = sum(bool((label.get("attributes") or {}).get("occluded")) for label in labels)
    truncated = sum(bool((label.get("attributes") or {}).get("truncated")) for label in labels)
    transition_count = 0
    for index in range(1, len(frames)):
        before = {
            str(label["id"]): bool((label.get("attributes") or {}).get("occluded"))
            for label in (frames[index - 1].get("labels") or [])
        }
        after = {
            str(label["id"]): bool((label.get("attributes") or {}).get("occluded"))
            for label in (frames[index].get("labels") or [])
        }
        transition_count += sum(before[key] or after[key] for key in before.keys() & after.keys())
    counts = [len(frame.get("labels") or []) for frame in frames]
    return {
        "labels": len(labels),
        "mean_objects_per_frame": float(np.mean(counts)),
        "maximum_objects_per_frame": int(max(counts, default=0)),
        "occluded_labels": int(occluded),
        "occluded_fraction": float(occluded / len(labels)) if labels else 0.0,
        "occlusion_transitions": int(transition_count),
        "truncated_fraction": float(truncated / len(labels)) if labels else 0.0,
    }


def rank(rows: list[dict], key, *, reverse: bool = True, limit: int = 20) -> list[dict]:
    return sorted(rows, key=key, reverse=reverse)[:limit]


def compact(row: dict) -> dict:
    return {
        "video_id": row["video_id"],
        "attributes": row["attributes"],
        "visual": row["visual"],
        "annotations": row["annotations"],
    }


def main() -> int:
    args = parse_args()
    image_root = DATA_ROOT / "images" / "track" / args.split
    truth_root = DATA_ROOT / "labels" / "box_track_20" / args.split
    image_labels = json.loads(
        (DATA_ROOT / f"labels/bdd100k_labels_images_{args.split}.json").read_text(
            encoding="utf-8"
        )
    )
    attributes = {Path(row["name"]).stem: row.get("attributes") or {} for row in image_labels}
    rows = []
    for index, directory in enumerate(sorted(path for path in image_root.iterdir() if path.is_dir()), 1):
        video_id = directory.name
        truth_path = truth_root / f"{video_id}.json"
        if not truth_path.is_file():
            continue
        frames = sorted(directory.glob("*.jpg"))
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "video_id": video_id,
                "attributes": attributes.get(video_id, {}),
                "visual": visual_statistics(sampled(frames, args.sample_frames)),
                "annotations": annotation_statistics(truth),
            }
        )
        if index % 50 == 0:
            print(f"[AUDIT] {index} видео", flush=True)

    daytime = [row for row in rows if row["attributes"].get("timeofday") == "daytime"]
    rankings = {
        "overexposure_proxy": [
            compact(row)
            for row in rank(
                daytime,
                lambda row: row["visual"]["highlight_fraction"],
                limit=args.top,
            )
        ],
        "low_light_proxy": [
            compact(row)
            for row in rank(
                rows,
                lambda row: row["visual"]["mean_luma"],
                reverse=False,
                limit=args.top,
            )
        ],
        "partial_occlusion": [
            compact(row)
            for row in rank(
                rows,
                lambda row: row["annotations"]["occlusion_transitions"],
                limit=args.top,
            )
        ],
        "complex_background_proxy": [
            compact(row)
            for row in rank(
                rows,
                lambda row: row["visual"]["edge_fraction"]
                * np.log1p(row["annotations"]["mean_objects_per_frame"]),
                limit=args.top,
            )
        ],
    }
    result = {
        "definitions": {
            "overexposure_proxy": "mean fraction of sampled grayscale pixels >= 248; not a BDD label",
            "low_light_proxy": "lowest mean grayscale luminance; not a BDD label",
            "partial_occlusion": "number of consecutive GT transitions touching an occluded label",
            "complex_background_proxy": "Canny edge fraction multiplied by log(1 + object density); not a BDD label",
        },
        "videos": len(rows),
        "sample_frames_per_video": args.sample_frames,
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: [row["video_id"] for row in value[:5]] for key, value in rankings.items()}, ensure_ascii=False, indent=2))
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
