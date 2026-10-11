"""Описательный разбор длинных разрывов аудита А1 (дополнение к карточке lost_link_audit_v1, только описание, без HOTA).

Те же разрывы, что в scripts/lost_link_audit.py (find_breaks на выходе после MASA, dev 35), в разрезах: класс объекта
разметки, ночь / не ночь, размер последней рамки старого номера (√площади: < 32, 32–96, ≥ 96 пикселей), пропуск
(31–60, 61–150, > 150 кадров). Для каждого разреза — число длинных разрывов, доля допустимых, доля в окне и частота на
1000 верных рамок разреза (знаменатель — верные рамки выхода того же класса / условия). Выход — runs/lost_link_audit/breakdown.json.

Запуск: python scripts/lost_link_breakdown.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lost_link_audit import LONG, SRC  # noqa: E402

OUT = PROJECT_ROOT / "runs/lost_link_audit/breakdown.json"


def size_bin(s):
    return "small<32" if s < 32 else ("medium32-96" if s < 96 else "large>=96")


def gap_bin(g):
    return "31-60" if g <= 60 else ("61-150" if g <= 150 else ">150")


def video(task):
    from scripts.chain_filter_head import classify_ids
    from scripts.lost_link_audit import find_breaks
    vid, root, split = task
    res = json.loads((SRC / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    cat = {str(l["id"]): l["category"] for fr in gt for l in fr.get("labels") or []}
    tp = Counter(cat[g] for tg in classify_ids(res, gt) for t, g in tg if t == "tp")
    breaks, _ = find_breaks(res, gt, vf)
    return vid, [{**b, "cls": cat[b["gt"]]} for b in breaks if b["long"]], dict(tp)


def main() -> int:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    split = dev[0][2]
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = set(filter_videos_by_subset(attrs, [v for v, _, _ in dev], "night"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(video, dev))
    rows, tp_cls, tp_cond = [], Counter(), Counter()
    for vid, bl, tp in got:
        cond = "night" if vid in night else "not_night"
        for b in bl:
            rows.append({**b, "cond": cond})
        tp_cls.update(tp); tp_cond[cond] += sum(tp.values())
    rep = {"long_breaks": len(rows), "videos": len(dev), "night_videos": len(night), "by": {}}
    dims = {"cls": lambda b: b["cls"], "cond": lambda b: b["cond"], "size": lambda b: size_bin(b["size"]), "gap": lambda b: gap_bin(b["gap"])}
    for name, key in dims.items():
        groups = {}
        for b in rows:
            groups.setdefault(key(b), []).append(b)
        out = {}
        for k, bs in sorted(groups.items(), key=lambda x: -len(x[1])):
            n = len(bs); d = {"long": n, "legal_share": round(sum(b["legal"] for b in bs) / n, 3), "window_share": round(sum(b["window"] for b in bs) / n, 3)}
            den = tp_cls.get(k) if name == "cls" else (tp_cond.get(k) if name == "cond" else None)
            if den:
                d["tp_boxes"] = int(den); d["per_1000_tp"] = round(1000 * n / den, 2)
            out[k] = d
        rep["by"][name] = out
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"[OK] {OUT} (длинный — пропуск > {LONG} кадров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
