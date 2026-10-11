"""Проверка механизма провала buffer_extend_v1: переходит ли номер на чужой объект после долгой потери (только описание).

Выход без сшивки (номера — номера трекера): буфер 30 (runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev) и буфер 150
(runs/buffer_extend/b150_none), dev 35. Для каждого номера выхода — последовательность его верных рамок (объект разметки,
кадр видео); «переход» — смена объекта разметки между соседними верными рамками одного номера; пропуск перехода — кадр
первой рамки нового объекта минус кадр последней рамки прежнего. Если провал вызван повторным захватом долго потерянного
трека, то лишние переходы буфера 150 должны приходиться на пропуски > 30 кадров (при буфере 30 таких быть не может:
потерянный трек удаляется через 30 кадров). Выход — runs/buffer_extend/hijack.json.

Запуск: python scripts/buffer_extend_hijack.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/buffer_extend"
DIRS = {"b30_none": PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev", "b150_none": OUT / "b150_none"}


def gap_bin(g):
    return "0" if g <= 1 else ("2-30" if g <= 30 else ("31-60" if g <= 60 else ("61-150" if g <= 150 else ">150")))


def transitions(task):
    from scripts.chain_filter_head import classify_ids
    d, vid, root, split = task
    res = json.loads((Path(d) / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    seq = {}
    for k, (fr, tg) in enumerate(zip(res, classify_ids(res, gt))):
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            if t == "tp":
                seq.setdefault(str(l["id"]), []).append((vf[k], g))
    gaps = []
    for s in seq.values():
        s.sort()
        gaps += [b[0] - a[0] for a, b in zip(s, s[1:]) if a[1] != b[1]]
    return gaps


def main() -> int:
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    rep = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, d in DIRS.items():
            gaps = [x for gs in pool.map(transitions, [(str(d), v, rt, sp) for v, rt, sp in dev]) for x in gs]
            c = Counter(gap_bin(x) for x in gaps)
            rep[name] = {"transitions": len(gaps), "by_gap": {k: c.get(k, 0) for k in ("0", "2-30", "31-60", "61-150", ">150")}}
    rep["extra_b150"] = {k: rep["b150_none"]["by_gap"][k] - rep["b30_none"]["by_gap"][k] for k in rep["b30_none"]["by_gap"]}
    OUT.joinpath("hijack.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"[OK] {OUT / 'hijack.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
