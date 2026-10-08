"""Эмбеддинги MASA-R50 для начала и конца каждой жизни трека (карточка stitch_appearance_models_v1).

tracking/masa_r50.py (без mmdetection; проверка: тот же трек через 5 кадров — сходство 0,955, разные — 0,103). Для кадров,
на которых начинается или кончается жизнь (runs/fly_signatures(_val)/<id>/signatures.npz: start/first_box, end/last_box),
считаются признаки всего кадра (длинная сторона 1024), эмбеддинг рамки — RoIAlign + голова QuasiDense (256), нормированный.
Сохраняется runs/masa_embeddings/<id>.npz (first, last) в порядке сигнатур.

Запуск: python scripts/build_masa_embeddings.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dino_embeddings import VAL7  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "masa_embeddings"


def main() -> int:
    import cv2
    import torch
    from tracking.masa_r50 import build, preprocess
    OUT.mkdir(parents=True, exist_ok=True)
    m = build()
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    jobs = [(v, PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "data/bdd100k/videos/train") for s in ("train", "dev", "test") for v in sel[s]]
    jobs += [(v, PROJECT_ROOT / "runs/fly_signatures_val", PROJECT_ROOT / "data/bdd100k/videos/val") for v in VAL7]
    t0 = time.time()
    for i, (v, sig, vdir) in enumerate(jobs):
        out = OUT / f"{v}.npz"
        if out.exists() or not (sig / v / "signatures.npz").exists():
            continue
        z = np.load(sig / v / "signatures.npz")
        need = {}
        for k in range(len(z["track_id"])):
            need.setdefault(int(z["start"][k]), []).append((k, 0, z["first_box"][k]))
            need.setdefault(int(z["end"][k]), []).append((k, 1, z["last_box"][k]))
        emb = np.zeros((len(z["track_id"]), 2, 256), np.float16)
        cap = cv2.VideoCapture(str(vdir / f"{v}.mov")); f = 0
        last_needed = max(need) if need else -1
        with torch.no_grad():
            while f <= last_needed:
                ok, fr = cap.read()
                if not ok:
                    break
                if f in need:
                    items = need[f]
                    x, s = preprocess(fr)
                    boxes = torch.tensor(np.stack([b for _, _, b in items]).astype(np.float32), device="cuda")
                    e = torch.nn.functional.normalize(m.embed_boxes(m.features(x), boxes, s), dim=1).cpu().numpy()
                    for (k, which, _), ee in zip(items, e):
                        emb[k, which] = ee
                f += 1
        cap.release()
        np.savez_compressed(out, first=emb[:, 0], last=emb[:, 1])
        print(f"[MASA] {i + 1}/{len(jobs)} {v}: жизней {len(emb)}, кадров с признаками {len(need)} | {(time.time() - t0) / 60:.1f} мин", flush=True)
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
