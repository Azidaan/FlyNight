"""Кэш поля движения мухи с нормализацией контраста перед глазом для видео val7 (карточка fly_input_norm_hota_v1).

Штатная муха 000, сетка плиток стенда 2×3, каждый кадр, вход lcn24 (scripts/fly_noise_probe.py: нормализация локального
контраста, σ = 24 px) — как в fly_input_norm_v1. Поле пишется с усилением 1 в формате кэша стенда:
runs/video_rate/<id>/motion/flyvis_tiled_lcn24.npz (points, vectors). Усиление 3,0 (выбрано по train в fly_input_norm_v1)
задаётся при подмешивании: scripts/evaluate_bytetrack_motion_injection.py --sources flyvis_tiled_lcn24 --gain-override 3.

Запуск: python scripts/build_fly_lcn_motion.py [--set dev]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fly_noise_probe import lcn  # noqa: E402
from scripts.fly_scale_probe import make_fly, video_field  # noqa: E402
from tracking.flyvis_geometry_protocol import add_geometry_arguments, ensure_geometry_output, resolve_output

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117",
        "b1d22449-15fb948f", "b232c7c9-d251d9ee", "b251064f-e7a165fd"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=("val7", "dev"), default="val7")
    add_geometry_arguments(ap); a = ap.parse_args()
    if a.set == "dev":   # 35 видео dev train_cache (motion_scheme_search_v1)
        vids = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]["dev"]
        root, vdir = PROJECT_ROOT / "runs/train_cache", PROJECT_ROOT / "data/bdd100k/videos/train"
    else:
        vids, root, vdir = VAL7, PROJECT_ROOT / "runs/video_rate", PROJECT_ROOT / "data/bdd100k/videos/val"
    cache_root = resolve_output(a.out, root, PROJECT_ROOT)
    for i, v in enumerate(vids):
        out = cache_root / v / "motion/flyvis_tiled_lcn24.npz"
        ensure_geometry_output(out, a.geometry_version, file=True)
        if out.exists():
            continue
        t0 = time.time()
        fps = json.loads((root / v / "motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
        fly = make_fly(fps, "2x3", 1, geometry_version=a.geometry_version)
        pts, vec = video_field(fly, vdir / f"{v}.mov", 1, lcn(24.0))
        np.savez_compressed(out, points=pts.astype(np.float32), vectors=vec.astype(np.float32), geometry_version=np.asarray(a.geometry_version))
        (out.with_suffix(".json")).write_text(json.dumps({"source": "flyvis_tiled_lcn24", "frame_count": int(len(vec)), "fps": fps, "flow_gain_in_cache": 1.0,
                                                          "input": "lcn24 (scripts/fly_noise_probe.py)", "layout": "2x3", "model": "flow/0000/000", "geometry_version": a.geometry_version}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[FIELD] {i + 1}/{len(vids)} {v}: кадров {len(vec)} | {time.time() - t0:.0f} с", flush=True)
        del fly
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
