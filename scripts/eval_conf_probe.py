"""Оценка пробы Г2 (карточка conf_target_probe_v1): полнота при равном FP, ночь/день/возраст 0, парно по сидам.

Для каждого кэша (исходный YOLO26 и каждое плечо_сид) — жадное сопоставление по
уверенности в каждом размеченном кадре (тот же класс, IoU ≥ 0,5, один к одному).
Бюджет FP — число FP исходного YOLO26 (runs/det_cache_ctl_960) на пороге 0,25 на
тех же 107 видео; порог каждой модели подбирается под этот бюджет. Разность
iou_free − ctl считается по каждому сиду и усредняется; ДИ — бутстреп по видео
(одна и та же выборка видео для всех сидов). mAP50 — последняя строка results.csv
(валидация Ultralytics на val step5 при 640).

Запуск: python scripts/eval_conf_probe.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.birth_cost import iou_mat, load_gt_tracks  # noqa: E402

BASE = PROJECT_ROOT / "runs" / "det_cache_ctl_960" / "val"
CACHES = PROJECT_ROOT / "runs" / "conf_probe_cache"
RUNS = PROJECT_ROOT / "runs" / "conf_probe"
OUT = RUNS / "eval.json"
SEEDS = (42, 123, 2024)


def conditions(stems):
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    cond = {}
    for c in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, sorted(stems), c):
            cond.setdefault(v, c)
    return cond


def video_scores(npz: Path, gt):
    """(score сопоставленного кандидата для каждого эталона или −inf, возраст эталона, score несопоставленных кандидатов)."""
    per_frame, age_of = gt
    d = np.load(npz)
    fidx, xyxy, conf, cls = d["frame_idx"], d["xyxy"], d["conf"], d["cls"].astype(np.int64)
    gs_all, ages_all, fp_all = [], [], []
    for fi in range(len(d["frame_names"])):
        sel = np.flatnonzero(fidx == fi)
        g = per_frame.get(fi)
        if g is None or len(g[0]) == 0:
            continue  # кадры без эталона не оцениваются (как в Г0б)
        gb, gc, tids = g
        gs = np.full(len(gb), -np.inf)
        used = np.zeros(len(gb), bool)
        m = np.where(gc[:, None] == cls[sel][None, :], iou_mat(gb, xyxy[sel]), 0.0) if len(sel) else np.zeros((len(gb), 0))
        for j in np.argsort(-conf[sel]):
            col = np.where(used, 0.0, m[:, j])
            k = int(col.argmax())
            if col[k] >= 0.5:
                used[k] = True; gs[k] = conf[sel][j]
            else:
                fp_all.append(conf[sel][j])
        gs_all.append(gs); ages_all.append(np.array([age_of[(t, fi)] for t in tids]))
    return np.concatenate(gs_all) if gs_all else np.zeros(0), np.concatenate(ages_all) if ages_all else np.zeros(0), np.array(fp_all, np.float32)


def main() -> int:
    stems = sorted(f.stem for f in BASE.glob("*.npz"))
    cond = conditions(stems)
    vids = [v for v in stems if v in cond]
    gts = {}
    for v in vids:
        per_frame, _, age_of = load_gt_tracks(v)
        gts[v] = (per_frame, age_of)
    models = {"base": BASE}
    for s in SEEDS:
        for a in ("ctl", "iou_free"):
            c = CACHES / f"{a}_s{s}" / "val"
            if c.exists() and len(list(c.glob("*.npz"))) >= len(stems):
                models[f"{a}_s{s}"] = c
    data = {m: {v: video_scores(p / f"{v}.npz", gts[v]) for v in vids} for m, p in models.items()}
    fp_budget = int(sum(int((data["base"][v][2] >= 0.25).sum()) for v in vids))
    thr = {}
    for m in models:
        fs = np.sort(np.concatenate([data[m][v][2] for v in vids]))[::-1]
        thr[m] = 0.25 if m == "base" else float(fs[min(fp_budget, len(fs)) - 1])
    # по видео: найдено, найдено возраста 0, эталонов, эталонов возраста 0
    cnt = {m: {v: (int((data[m][v][0] >= thr[m]).sum()), int(((data[m][v][0] >= thr[m]) & (data[m][v][1] == 0)).sum())) for v in vids} for m in models}
    n = {v: (len(data["base"][v][0]), int((data["base"][v][1] == 0).sum())) for v in vids}
    groups = {"night": [v for v in vids if cond[v] == "night"], "day": [v for v in vids if cond[v] == "normal"], "all": vids}

    def rec(m, vs, age0=False):
        i = 1 if age0 else 0
        return sum(cnt[m][v][i] for v in vs) / max(sum(n[v][i] for v in vs), 1)
    res = {"fp_budget": fp_budget, "thresholds": thr, "videos": len(vids), "recall": {}, "diff": {}, "ci95": {}, "map50": {}}
    for m in models:
        res["recall"][m] = {g: rec(m, vs) for g, vs in groups.items()} | {f"{g}_age0": rec(m, vs, True) for g, vs in groups.items()}
        rc = RUNS / m / "results.csv"
        if rc.exists():
            rows = list(csv.DictReader(rc.open(encoding="utf-8")))
            res["map50"][m] = float(rows[-1]["metrics/mAP50(B)"].strip()) if rows else None
    paired = [s for s in SEEDS if f"ctl_s{s}" in models and f"iou_free_s{s}" in models]
    res["paired_seeds"] = paired
    rng = np.random.default_rng(0)
    for g, vs in groups.items():
        for age0 in (False, True):
            key = f"{g}_age0" if age0 else g
            if not paired:
                continue
            res["diff"][key] = float(np.mean([rec(f"iou_free_s{s}", vs, age0) - rec(f"ctl_s{s}", vs, age0) for s in paired]))
            boots = []
            for _ in range(2000):
                pick = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
                boots.append(np.mean([rec(f"iou_free_s{s}", pick, age0) - rec(f"ctl_s{s}", pick, age0) for s in paired]))
            res["ci95"][key] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            ctl = [rec(f"ctl_s{s}", vs, age0) for s in paired]
            res.setdefault("ctl_seed_range", {})[key] = float(max(ctl) - min(ctl))
    if paired and all(f"{a}_s{s}" in res["map50"] for s in paired for a in ("ctl", "iou_free")):
        res["diff"]["map50"] = float(np.mean([res["map50"][f"iou_free_s{s}"] - res["map50"][f"ctl_s{s}"] for s in paired]))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"бюджет FP {fp_budget}; видео {len(vids)}; парных сидов {len(paired)}")
    for m in models:
        r = res["recall"][m]
        print(f"  {m:16s} порог {thr[m]:.4f} ночь {r['night']:.4f} день {r['day']:.4f} возраст0 ночь {r['night_age0']:.4f} mAP50 {res['map50'].get(m)}")
    for k, v in res["diff"].items():
        print(f"  iou_free − ctl {k}: {v:+.4f} {res['ci95'].get(k, '')} размах ctl {res.get('ctl_seed_range', {}).get(k, '')}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
