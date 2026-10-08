"""Сопоставление рамок готового выхода с записями жизней (исправление 09.10, life_filter_oracle_v2 / life_filter_head_v2).

Один номер трекера бывает у нескольких жизней: трек потерян и вернулся с прежним номером — в signatures.npz это
несколько записей с одним track_id и разными интервалами [start, end]. Уникальный идентификатор жизни — номер записи
в signatures.npz. Рамка выхода (опорный кадр i → кадр видео f = alignment.matches[i].video_frame_index) относится:
  1) к строке выхода трекера кадра f с совпадающими координатами (допуск 0,01 пикселя) → номер трекера;
  2) к записи жизни с этим номером, у которой start ≤ f ≤ end.
Неоднозначные случаи (нет строки / несколько строк / нет записи / несколько записей) — None и отдельный счёт.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def box_segments(vid: str, sig_dir: str, root: str, res: list) -> tuple[list, dict]:
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    z = np.load(Path(sig_dir) / vid / "signatures.npz")
    o = z["outputs"]; tids = z["track_id"].astype(int); st = z["start"].astype(int); en = z["end"].astype(int)
    by = {}
    for r in o:
        by.setdefault(int(r[0]), []).append(r)
    recs = {}
    for i, t in enumerate(tids):
        recs.setdefault(int(t), []).append(i)
    stats = {"boxes": 0, "ok": 0, "no_row": 0, "multi_row": 0, "no_segment": 0, "multi_segment": 0}
    seg = []
    for k, fr in enumerate(res):
        f = int(matches[k]["video_frame_index"]); rows = by.get(f, []); cur = []
        for l in fr.get("labels") or []:
            stats["boxes"] += 1
            b = l["box2d"]; key = np.array([b["x1"], b["y1"], b["x2"], b["y2"]])
            hit = [int(r[5]) for r in rows if np.allclose(r[1:5], key, atol=1e-2)]
            if len(hit) != 1:
                stats["no_row" if not hit else "multi_row"] += 1; cur.append(None); continue
            cand = [i for i in recs.get(hit[0], []) if st[i] <= f <= en[i]]
            if len(cand) != 1:
                stats["no_segment" if not cand else "multi_segment"] += 1; cur.append(None); continue
            stats["ok"] += 1; cur.append(cand[0])
        seg.append(cur)
    return seg, stats
