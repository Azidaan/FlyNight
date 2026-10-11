"""Что изменил буфер 150 (дополнение к карточке buffer_extend_v1, только описание, без HOTA).

Выходы после MASA на dev: буфер 30 (runs/chain_filter_rank/repro/dev) и буфер 150 (runs/buffer_extend/b150_masa); то же без
сшивки (b30_none / b150_none). Для каждого выхода:
  разрывы номера объектов разметки (scripts/lost_link_audit.find_breaks) по пропуску (≤ 30, 31–150, > 150 кадров);
  ложные слияния — номер выхода, чьи верные рамки принадлежат двум и более объектам разметки (число таких номеров и
  число «лишних» объектов: сумма (объектов на номер − 1));
  верные рамки (tp) и рамки без пары (fp), по правилу оценщика (classify_ids).
По видео, ночь отдельно. Выход — runs/buffer_extend/diag.json.

Запуск: python scripts/buffer_extend_diag.py
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
DIRS = {"b30_masa": PROJECT_ROOT / "runs/chain_filter_rank/repro/dev", "b150_masa": OUT / "b150_masa",
        "b30_none": PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev", "b150_none": OUT / "b150_none"}


def gap_bin(g):
    return "le30" if g <= 30 else ("31-150" if g <= 150 else "gt150")


def stats(task):
    from scripts.chain_filter_head import classify_ids
    from scripts.lost_link_audit import find_breaks
    d, vid, root, split = task
    res = json.loads((Path(d) / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    tags = classify_ids(res, gt)
    owners, kinds = {}, Counter()
    for fr, tg in zip(res, tags):
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            kinds[t] += 1
            if t == "tp":
                owners.setdefault(str(l["id"]), set()).add(g)
    breaks, _ = find_breaks(res, gt, vf)
    s = Counter(gap_bin(b["gap"]) for b in breaks)
    return vid, {"breaks": len(breaks), **{f"breaks_{k}": s.get(k, 0) for k in ("le30", "31-150", "gt150")},
                 "merged_ids": sum(1 for o in owners.values() if len(o) > 1), "extra_objects": sum(len(o) - 1 for o in owners.values()),
                 "ids": len(owners), "tp": kinds.get("tp", 0), "fp": kinds.get("fp", 0)}


def main() -> int:
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"] if (DIRS["b150_masa"] / f"{v}.json").exists()]
    split = dev[0][2]; vids = [v for v, _, _ in dev]
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = set(filter_videos_by_subset(attrs, vids, "night"))
    per = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, d in DIRS.items():
            per[name] = dict(pool.map(stats, [(str(d), v, rt, sp) for v, rt, sp in dev]))
    rep = {"videos": len(vids), "night_videos": len(night), "total": {}, "per_video": per}
    for g, gv in (("all", vids), ("night", [v for v in vids if v in night]), ("not_night", [v for v in vids if v not in night])):
        for name in DIRS:
            keys = per[name][vids[0]].keys()
            rep["total"][f"{name}|{g}"] = {k: int(sum(per[name][v][k] for v in gv)) for k in keys}
    OUT.joinpath("diag.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in rep["total"].items():
        print(k, v, flush=True)
    print(f"[OK] {OUT / 'diag.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
