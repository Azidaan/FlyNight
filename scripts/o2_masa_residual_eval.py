"""Остаток запаса оракула O2 после сшивки MASA (карточка o2_masa_residual_v1, проверка перед З13).

fovea_target_30hz_v1: оракул O2 (своя уверенная детекция в области глаза, по разметке) дал +0,47 HOTA без сшивки. Здесь —
тот же трекер с оракулом (scripts/build_fly_signatures.py --oracle O2 → runs/fly_signatures_o2; эмбеддинги —
runs/dino_embeddings_o2, runs/masa_embeddings_o2), затем та же сшивка «база + MASA» (окно 150; 5, порог 0,6; голова пары —
обученная на парах train 130 стенда, как в stitch_eval_v2 и buffer_extend_eval).
Сравнение на dev 35: после MASA с O2 против после MASA стенда (runs/chain_filter_rank/repro/dev); сверка — O2 без сшивки
против 61,807 (runs/fovea_target_30hz/report.json). Критерий (до прогона): общий прирост ≥ +0,3 и ДИ по видео > 0.

Запуск: python scripts/o2_masa_residual_eval.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402
from scripts.buffer_extend_eval import KEYS, long_breaks, per_video_hota  # noqa: E402

SIG = PROJECT_ROOT / "runs/fly_signatures_o2"
DINO, MASA = PROJECT_ROOT / "runs/dino_embeddings_o2", PROJECT_ROOT / "runs/masa_embeddings_o2"
OUT = PROJECT_ROOT / "runs/o2_masa_residual"
STAND_MASA = PROJECT_ROOT / "runs/chain_filter_rank/repro/dev"
STAND_NONE = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev"
O2_REF = 61.80698189906935   # runs/fovea_target_30hz/report.json, pooled O2|all


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"] if (SIG / v / "signatures.npz").exists() and (MASA / f"{v}.npz").exists()]
    vids = [v for v, _, _ in dev]; split = dev[0][2]; root = dev[0][1]
    print(f"[PLAN] dev с оракулом O2: {len(vids)} видео", flush=True)
    with ProcessPoolExecutor(max_workers=4, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    with ProcessPoolExecutor(max_workers=4, initializer=V._set_dirs, initargs=(150, 5.0, str(DINO), str(MASA))) as pool:
        pairs = list(pool.map(V.video_pairs, [(v, str(SIG), root, split) for v in vids]))
    for thr, tag in ((0.6, "masa"), (2.0, "none")):
        tasks = [(d["video_id"], str(SIG), root, split, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), thr, str(OUT / f"o2_{tag}")) for d in pairs]
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(stitch_video, tasks))
    print("[STITCH] O2: сшитый выход и без сшивки готовы", flush=True)
    dirs = {"stand_masa": STAND_MASA, "o2_masa": OUT / "o2_masa", "stand_none": STAND_NONE, "o2_none": OUT / "o2_none"}
    per = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, d in dirs.items():
            per[name] = dict(pool.map(per_video_hota, [(str(d), v, split) for v in vids]))
        lb = {name: {v: (n, m) for v, n, m in pool.map(long_breaks, [(str(dirs[name]), v, root, split) for v in vids])} for name in ("stand_masa", "o2_masa")}
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))
    ev = {}
    for v in vids:
        for k, x in json.loads((SIG / v / "signatures.json").read_text(encoding="utf-8")).get("oracle_events", {}).items():
            ev[k] = ev.get(k, 0) + int(x)
    rep = {"videos": len(vids), "pooled": {}, "diff": {}, "oracle_events": ev,
           "breaks": {k: {"long": int(sum(x[0] for x in v.values())), "all": int(sum(x[1] for x in v.values()))} for k, v in lb.items()}}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", night)):
        for name, d in dirs.items():
            m = evaluate_hota(d, gv, split=split); rep["pooled"][f"{name}|{g}"] = {k: m.get(k) for k in KEYS}
        for a_, b_ in (("o2_masa", "stand_masa"), ("o2_none", "stand_none"), ("stand_masa", "stand_none"), ("o2_masa", "o2_none")):
            for met in ("HOTA", "AssA", "DetA"):
                dd = np.array([per[a_][v][met] - per[b_][v][met] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                rep["diff"][f"{a_}-{b_}|{g}|{met}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "n": len(dd),
                                                     "pooled_gain": rep["pooled"][f"{a_}|{g}"][met] - rep["pooled"][f"{b_}|{g}"][met]}
    rep["repro_o2_none"] = {"got": rep["pooled"]["o2_none|all"]["HOTA"], "ref": O2_REF, "ok": bool(abs(rep["pooled"]["o2_none|all"]["HOTA"] - O2_REF) <= 0.01)}
    d = rep["diff"]["o2_masa-stand_masa|all|HOTA"]
    rep["criterion_met"] = bool(rep["repro_o2_none"]["ok"] and d["pooled_gain"] >= 0.3 and d["ci95"][0] > 0)
    for g in ("all", "night"):
        print(f"[HOTA] {g}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{g}']['HOTA']:.3f} (AssA {rep['pooled'][f'{n}|{g}']['AssA']:.2f}, IDSw {rep['pooled'][f'{n}|{g}']['IDSw']:.0f})" for n in dirs), flush=True)
        for k, x in rep["diff"].items():
            if f"|{g}|HOTA" in k:
                print(f"[DIFF] {k}: общий {x['pooled_gain']:+.3f}; по видео {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[REPRO] {rep['repro_o2_none']}\n[EVENTS] {ev}\n[BREAKS] {rep['breaks']}\n[CRITERION] {rep['criterion_met']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
