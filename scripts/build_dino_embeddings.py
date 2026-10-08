"""Эмбеддинги внешности DINOv3 ViT-S/16 для начала и конца каждой жизни трека (карточка stitch_dino_v1).

Замена узла «внешность» в голове узнавания (сейчас — косинус миниатюры и гистограмма). Модель — timm
vit_small_patch16_dinov3.lvd1689m (веса — weights/dinov3/, лицензия DINOv3, Meta; открытое зеркало timm), без дообучения.
Для каждой жизни из runs/fly_signatures(_val)/<id>/signatures.npz: вырез рамки first_box на кадре start и last_box на кадре
end (рамка расширена на 10 %, вписана в квадрат с отступом, 128×128, нормировка ImageNet), выход — средний по токенам
признак 384, нормированный. Сохраняется runs/dino_embeddings/<id>.npz (first, last) в порядке сигнатур.

Запуск: python scripts/build_dino_embeddings.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "dino_embeddings"
WEIGHTS = PROJECT_ROOT / "weights/dinov3/vit_small_patch16_dinov3.lvd1689m.safetensors"
SIZE = 128
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]


def crop(frame, box):
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(np.float64)
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 4) * 1.1, max(y2 - y1, 4) * 1.1
    s = max(bw, bh)
    X1, Y1, X2, Y2 = int(round(cx - s / 2)), int(round(cy - s / 2)), int(round(cx + s / 2)), int(round(cy + s / 2))
    pad = cv2.copyMakeBorder(frame, max(0, -Y1), max(0, Y2 - h), max(0, -X1), max(0, X2 - w), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    ox, oy = max(0, -X1), max(0, -Y1)
    c = pad[Y1 + oy:Y2 + oy, X1 + ox:X2 + ox]
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def main() -> int:
    import cv2
    import timm
    import torch
    from safetensors.torch import load_file
    OUT.mkdir(parents=True, exist_ok=True)
    m = timm.create_model("vit_small_patch16_dinov3", pretrained=False, num_classes=0)
    m.load_state_dict(load_file(str(WEIGHTS)), strict=True); m = m.eval().cuda().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
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
        emb = np.zeros((len(z["track_id"]), 2, 384), np.float16)
        cap = cv2.VideoCapture(str(vdir / f"{v}.mov")); f = 0; batch, idx = [], []

        def flush():
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).half() / 255.0
            x = ((x[:, [2, 1, 0]] - mean) / std).half()   # BGR → RGB
            with torch.no_grad():
                y = m(x).float(); y = torch.nn.functional.normalize(y, dim=1).cpu().numpy()
            for (k, which), e in zip(idx, y):
                emb[k, which] = e
            batch.clear(); idx.clear()
        last_needed = max(need) if need else -1
        while f <= last_needed:
            ok, fr = cap.read()
            if not ok:
                break
            for k, which, box in need.get(f, []):
                batch.append(crop(fr, box)); idx.append((k, which))
            if len(batch) >= 256:
                flush()
            f += 1
        flush(); cap.release()
        np.savez_compressed(out, first=emb[:, 0], last=emb[:, 1])
        el = time.time() - t0
        print(f"[DINO] {i + 1}/{len(jobs)} {v}: жизней {len(emb)} | {el / 60:.1f} мин", flush=True)
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
