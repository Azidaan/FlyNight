"""Кэши новых val-видео для проверки life_filter_newval_v1: выравнивание + детекции стенда + частота кадров.

Повторяет ступень кэшей train (scripts/build_train_caches.py) без DIS и фовеального потока: сигнатуры мухи
(scripts/build_fly_signatures.py) читают только детекции (detections/raw_video.npz) и частоту из motion/flyvis_tiled.json.
Детекции — YOLO26s (weights/baseline_seed42/best.pt), 960, conf 0,01, iou 0,7, как в стенде. Видео — из
runs/newval_protocol/videos.json (night + normal), только с alignment_ok в runs/newval_videos/manifest.json.
Повторный запуск дополняет.

Запуск: python scripts/newval_build.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import build_detection_cache, save_json  # noqa: E402

WEIGHTS = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"
VIDEOS = PROJECT_ROOT / "data/bdd100k/videos/val"
ALIGN = PROJECT_ROOT / "runs/newval_videos"
OUT = PROJECT_ROOT / "runs/newval_cache"


def main() -> int:
    spec = json.loads((PROJECT_ROOT / "runs/newval_protocol/videos.json").read_text(encoding="utf-8"))
    manifest = json.loads((ALIGN / "manifest.json").read_text(encoding="utf-8"))["videos"]
    ids = [v for v in spec["night"] + spec["normal"] if manifest.get(v, {}).get("alignment_ok")]
    print(f"[PLAN] видео с выравниванием {len(ids)} из {len(spec['night']) + len(spec['normal'])}", flush=True)
    t0 = time.perf_counter()
    for k, v in enumerate(ids, 1):
        out = OUT / v; out.mkdir(parents=True, exist_ok=True)
        if not (out / "alignment.json").exists():
            shutil.copyfile(ALIGN / v / "alignment.json", out / "alignment.json")
        fps = float(manifest[v]["container"]["fps"])
        meta = out / "motion" / "flyvis_tiled.json"
        if not meta.exists():
            meta.parent.mkdir(parents=True, exist_ok=True)
            save_json(meta, {"source": "fps_only", "fps": fps, "note": "только частота для build_fly_signatures (как в build_train_caches)"})
        build_detection_cache(VIDEOS / f"{v}.mov", WEIGHTS, out / "detections" / "raw_video.npz", imgsz=960, conf=0.01, iou=0.7, device="0", force=False)
        el = time.perf_counter() - t0
        print(f"[CACHE] {k}/{len(ids)} {v}: всего {el / 60:.1f} мин, ожидается ещё {el / k * (len(ids) - k) / 60:.1f} мин", flush=True)
    save_json(OUT / "selection.json", {"videos": ids, "night": [v for v in spec["night"] if v in ids], "normal": [v for v in spec["normal"] if v in ids]})
    print("[OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
