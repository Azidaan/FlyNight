"""Ступень 1 кэшей на обучающих видео: детекции YOLO26, DIS (четверть), фовеальный поток FlyVis.

Параметры детектора и трекера — из замороженного стенда (bench_freeze_v1).
Выбор роликов детерминирован: первые N по sha1(ID) внутри каждого сплита
манифеста цели 1а (только alignment_ok). Повторный запуск дополняет.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import build_detection_cache, build_motion_cache, save_json  # noqa: E402

WEIGHTS = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "runs" / "train_videos" / "manifest.json")
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "train")
    parser.add_argument("--alignments", type=Path, default=PROJECT_ROOT / "runs" / "train_videos")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--per-split", default="train=200,dev=50,test=50")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-fovea", action="store_true")
    return parser.parse_args()


def select_videos(manifest: dict, per_split: str) -> dict[str, list[str]]:
    quota = {k: int(v) for k, v in (item.split("=") for item in per_split.split(","))}
    chosen: dict[str, list[str]] = {}
    for split, count in quota.items():
        ids = [v for v, r in manifest["videos"].items() if r.get("alignment_ok") and r["split"] == split]
        ids.sort(key=lambda v: hashlib.sha1(v.encode()).hexdigest())
        chosen[split] = ids[:count]
    return chosen


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chosen = select_videos(manifest, args.per_split)
    order = [(split, v) for split in ("train", "dev", "test") for v in chosen.get(split, [])]
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "selection.json", {"per_split": args.per_split, "rule": "первые N по sha1(ID) среди alignment_ok", "videos": chosen})
    print(f"[PLAN] {len(order)} роликов: " + ", ".join(f"{s} {len(v)}" for s, v in chosen.items()), flush=True)
    started = time.perf_counter()
    for index, (split, video_id) in enumerate(order, 1):
        out = args.output / video_id
        out.mkdir(parents=True, exist_ok=True)
        video = args.videos / f"{video_id}.mov"
        alignment_src = args.alignments / video_id / "alignment.json"
        if not (out / "alignment.json").exists():
            shutil.copyfile(alignment_src, out / "alignment.json")
        fps = float(manifest["videos"][video_id]["container"]["fps"])
        tick = time.perf_counter()
        build_detection_cache(video, WEIGHTS, out / "detections" / "raw_video.npz", imgsz=960, conf=0.01, iou=0.7, device=args.device, force=False)
        build_motion_cache("dis_quarter", video, out / "motion", fps=fps, max_dt=0.02, force=False)
        if not args.skip_fovea and not (out / "fovea_flow.npz").exists():
            # Сборщик фовеального потока читает fps из motion/flyvis_tiled.json — подложим метаданные с частотой.
            fps_meta = out / "motion" / "flyvis_tiled.json"
            if not fps_meta.exists():
                save_json(fps_meta, {"source": "fps_only", "fps": fps, "note": "только частота для build_foveated_readout_cache --decode-flow"})
            subprocess.run([sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "build_foveated_readout_cache.py"), "--decode-flow", "--video-id", video_id,
                            "--videos", str(args.videos), "--detections-root", str(args.output), "--flow-root", str(args.output), "--output", str(args.output)], check=True)
        elapsed = time.perf_counter() - started
        print(f"[CACHE] {index}/{len(order)} {split} {video_id}: {time.perf_counter() - tick:.0f} с; всего {elapsed / 3600:.2f} ч, ожидается ещё {(elapsed / index) * (len(order) - index) / 3600:.2f} ч", flush=True)
    print(f"[OK] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
