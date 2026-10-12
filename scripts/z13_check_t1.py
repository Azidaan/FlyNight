"""З13, техническая проверка Т1 (процессор): данные и координаты (карточка fly_object_readout_z13_v1).

По сохранённым эпизодам runs/z13_data/<id>.npz:
  * перевод «центр разметки → колонка → точка кадра»: расстояние до центра в пикселях поля глаза < шага колонки (все положительные);
  * метки заново выводятся из исходной разметки (scripts/z13_data.label_frame по сохранённым вырезкам и объекту образца) — совпадение 100 %;
  * рамка образца даёт непустую маску колонок; кадры эпизода идут подряд;
  * 20 положительных примеров картинкой: вход глаза на кадре поиска (колонки), цель-колонка, центр разметки; образец с маской.
Выход — runs/z13_data/t1.json, runs/z13_data/t1.png.

Запуск: python scripts/z13_check_t1.py [--data runs/z13_data] [--videos N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.cuda.is_available = lambda: False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.output_filter_oracle import VALID
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.z13_data import label_frame, raw_index
    from tracking.foveated_eyes import EyeBank
    from tracking.z13_readout import N_EXEMPLAR, HexGeometry, to_eye, to_frame
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "runs/z13_data"); ap.add_argument("--videos", type=int, default=10); a = ap.parse_args()
    bank = EyeBank(make_motion_estimator("flyvis_tiled", 30.0, 0.02)); es = bank.eye_size
    geom = HexGeometry(bank.points_local, es)
    files = sorted(a.data.glob("*.npz"))[:a.videos]
    rep = {"videos": len(files), "spacing_eye_px": geom.spacing, "roundtrip_max_eye_px": 0.0, "positives": 0, "labels": 0, "label_mismatch": 0, "empty_masks": 0, "nonconsecutive_search": 0}
    examples = []
    for fp in files:
        vid = fp.stem; d = np.load(fp)
        vf = [int(m["video_frame_index"]) for m in json.loads((PROJECT_ROOT / "runs/train_cache" / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
        R = raw_index(json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train" / f"{vid}.json").read_text(encoding="utf-8")), vf, VALID)
        ann = sorted(R); nb = {f: (ann[i - 1] if i else None, ann[i + 1] if i + 1 < len(ann) else None) for i, f in enumerate(ann)}
        off, fr = d["ep_offset"], d["fr_frame"]
        rep["empty_masks"] += int((~d["ex_mask"].any(1)).sum())
        for e in range(len(off) - 1):
            s = fr[off[e] + N_EXEMPLAR:off[e + 1]]
            rep["nonconsecutive_search"] += int(len(s) > 1 and np.any(np.diff(s) != 1))
        for i in range(len(d["lab_kind"])):
            e = int(d["lab_ep"][i]); f = int(fr[off[e] + N_EXEMPLAR + d["lab_pos"][i]]); crop = tuple(int(v) for v in d["lab_crop"][i])
            g = str(d["ep_identity"][e]) or None
            a_, b_ = nb[f]
            around = g is not None and a_ is not None and b_ is not None and any(o[0] == g for o in R[a_]) and any(o[0] == g for o in R[b_])
            kind, col, c, b = label_frame(g, int(d["ep_id_dist"][e]), R[f], crop, geom, es, around)
            rep["labels"] += 1; rep["label_mismatch"] += int(kind != int(d["lab_kind"][i]) or col != int(d["lab_col"][i]))
            if kind == 1:
                rep["positives"] += 1
                pe = geom.points[col]; ce = np.asarray(to_eye(c, crop, es))
                rep["roundtrip_max_eye_px"] = max(rep["roundtrip_max_eye_px"], float(np.linalg.norm(pe - ce)))
                back = np.asarray(to_frame(ce, crop, es)); assert np.allclose(back, c, atol=1e-3)
                if len(examples) < 20 and np.random.default_rng(len(examples) + i).random() < 0.05:
                    examples.append((vid, d["fr_hex"][off[e] + N_EXEMPLAR + d["lab_pos"][i]].astype(np.float32), col, ce, d["fr_hex"][off[e] + N_EXEMPLAR - 1].astype(np.float32), d["ex_mask"][e]))
    rep["passed"] = bool(rep["label_mismatch"] == 0 and rep["roundtrip_max_eye_px"] < geom.spacing and rep["empty_masks"] == 0 and rep["nonconsecutive_search"] == 0)
    fig, axes = plt.subplots(5, 8, figsize=(20, 12.5))
    for k, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        j = k // 2
        if j >= len(examples):
            continue
        vid, hs, col, ce, hx, mask = examples[j]
        if k % 2 == 0:
            ax.scatter(geom.points[:, 0], -geom.points[:, 1], c=hs, s=6, cmap="gray"); ax.scatter([geom.points[col, 0]], [-geom.points[col, 1]], s=60, facecolors="none", edgecolors="r")
            ax.plot([ce[0]], [-ce[1]], "c+", ms=10); ax.set_title(f"поиск {vid[:8]}", fontsize=7)
        else:
            ax.scatter(geom.points[:, 0], -geom.points[:, 1], c=hx, s=6, cmap="gray"); ax.scatter(geom.points[mask, 0], -geom.points[mask, 1], s=3, c="y"); ax.set_title("образец, маска", fontsize=7)
    fig.tight_layout(); fig.savefig(a.data / "t1.png", dpi=80)
    (a.data / "t1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[T1] {rep}\n[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
