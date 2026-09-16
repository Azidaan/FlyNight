"""Цель 1а: манифест исходных обучающих видео BDD100K MOT и фиксированное разбиение.

Для каждого скачанного ролика: sha256 и размер (из манифеста загрузки), число
кадров и частота по контейнеру, читаемость, сопоставление размеченных кадров с
исходными по содержимому (tracking.video_alignment, как для семи val-видео),
аудит качества выравнивания. Разбиение train/dev/test — детерминированно по
хэшу ID видео, фиксируется до любого обучения. Повторный запуск дополняет.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import build_alignment, save_json  # noqa: E402

MAX_MATCH_MAE = 2.5      # у семи val-видео максимум 1,29
MAX_STEP = 12            # шаг между опорными кадрами (у val 1–10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "images" / "track")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "train_videos")
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def assign_split(video_id: str) -> str:
    """70/15/15 по sha1(ID): детерминированно, независимо от порядка загрузки."""
    bucket = int(hashlib.sha1(video_id.encode("ascii")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else ("dev" if bucket < 85 else "test")


def main() -> int:
    args = parse_args()
    fetch_manifest = json.loads((args.videos / f"{args.split}_manifest.json").read_text(encoding="utf-8"))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"split_rule": "sha1(id)[:8] % 100: <70 train, <85 dev, else test", "videos": {}}
    ids = sorted(v for v in fetch_manifest["videos"] if (args.videos / args.split / f"{v}.mov").exists())
    if args.limit:
        ids = ids[: args.limit]
    started = time.perf_counter()
    for index, video_id in enumerate(ids, 1):
        if video_id in manifest["videos"] and manifest["videos"][video_id].get("alignment_ok") is not None:
            continue
        video = args.videos / args.split / f"{video_id}.mov"
        images = sorted((args.images / args.split / video_id).glob("*.jpg"))
        labels = json.loads((args.labels / args.split / f"{video_id}.json").read_text(encoding="utf-8"))
        record = {"sha256": fetch_manifest["videos"][video_id]["sha256"], "bytes": fetch_manifest["videos"][video_id]["bytes"], "container": {k: fetch_manifest["videos"][video_id].get(k) for k in ("frames", "fps", "width", "height", "opened")},
                  "labelled_frames": len(labels), "reference_images": len(images), "split": assign_split(video_id)}
        try:
            rows, metadata = build_alignment(video, images, out / video_id / "alignment.json", args.search_radius)
            errors = [r["mean_absolute_error"] for r in rows]
            steps = np.diff([r["video_frame_index"] for r in rows]) if len(rows) > 1 else np.asarray([])
            record.update({"decoded_frames": metadata.get("frame_count", metadata.get("frames")), "alignment": {"max_mae": float(max(errors)), "median_mae": float(np.median(errors)), "max_step": int(steps.max()) if steps.size else None, "min_step": int(steps.min()) if steps.size else None, "monotone": bool((steps > 0).all()) if steps.size else True},
                           "alignment_ok": bool(len(rows) == len(labels) == len(images) and max(errors) <= MAX_MATCH_MAE and (steps.size == 0 or ((steps > 0).all() and steps.max() <= MAX_STEP)))})
        except Exception as error:  # noqa: BLE001 — ролик не должен ронять весь манифест
            record.update({"alignment_ok": False, "error": str(error)})
        manifest["videos"][video_id] = record
        if index % 20 == 0 or index == len(ids):
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            ok = sum(1 for v in manifest["videos"].values() if v.get("alignment_ok"))
            print(f"[1A] {index}/{len(ids)} {video_id}: ok {ok}/{len(manifest['videos'])}, {(time.perf_counter() - started) / 60:.1f} мин", flush=True)
    counts = {s: sum(1 for v in manifest["videos"].values() if v["split"] == s) for s in ("train", "dev", "test")}
    manifest["summary"] = {"videos": len(manifest["videos"]), "alignment_ok": sum(1 for v in manifest["videos"].values() if v.get("alignment_ok")), "split_counts": counts,
                           "fps_range": [min(v["container"]["fps"] for v in manifest["videos"].values()), max(v["container"]["fps"] for v in manifest["videos"].values())] if manifest["videos"] else None,
                           "failed": [k for k, v in manifest["videos"].items() if not v.get("alignment_ok")]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False)[:600])
    print(f"[OK] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
