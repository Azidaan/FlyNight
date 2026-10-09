"""Проверка фильтра жизней на новых val-видео (карточка life_filter_newval_v1). Всё зафиксировано заранее.

Конвейер на видео из runs/newval_cache/selection.json (ночь — основная группа, день — дополнительная):
  1) кандидаты сшивки (scripts/stitch_eval_v2.video_pairs, окно 150; 5; эмбеддинги — runs/dino_embeddings_newval,
     runs/masa_embeddings_newval), голова пары «база + MASA», обученная на парах train 130 с меткой (как в stitch_eval_v2),
     жадная сшивка с порогом 0,6 (scripts/eval_fly_reid.stitch_video) → runs/newval_eval/stitched;
  2) жизни (запись = номер + интервал) и признаки — scripts/life_filter_head.video_task (режим v2);
  3) фильтр: итоговые модели runs/life_filter_rule_cv/final_nofly.pkl и final_fly.pkl, пороги из life_filter_rule_cv_v1 —
     без мухи 0,85, с мухой 0,70; дополнительно общий порог 0,70 для обеих голов; удаление записей жизней
     (scripts/life_filter_head.apply_task);
  4) HOTA, DetA, AssA, CLR_TP/FP/FN (TrackEval) — общие по группе и по видео; удалённые ложные/верные рамки, удалённые
     жизни по типу (полностью ложные / смешанные / настоящие / без учитываемых), потерянные объекты — по группам.
Критерии (до данных): ночь — HOTA «фильтр с мухой (0,70) − фильтр без мухи (0,85)» по видео > 0, ДИ > 0 (бутстрэп 5000);
при общем пороге 0,70 — тот же знак; фильтр с мухой − без фильтра > 0; потери — объекты ≤ 1 %, верные рамки ≤ 0,5 %.
День — те же величины, дополнительно. Пороги не меняются.

Запуск: python scripts/newval_eval.py
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402
from scripts.life_filter_head import KEYS, apply_task, losses, video_task  # noqa: E402

CACHE = PROJECT_ROOT / "runs/newval_cache"
SIG = PROJECT_ROOT / "runs/fly_signatures_newval"
OUT = PROJECT_ROOT / "runs/newval_eval"
DINO, MASA = PROJECT_ROOT / "runs/dino_embeddings_newval", PROJECT_ROOT / "runs/masa_embeddings_newval"
RULE = PROJECT_ROOT / "runs/life_filter_rule_cv"
VARIANTS = {"none": (None, None), "nofly@0.85": ("nofly", 0.85), "fly@0.7": ("fly", 0.7), "nofly@0.7": ("nofly", 0.7)}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((CACHE / "selection.json").read_text(encoding="utf-8"))
    vids = [v for v in sel["videos"] if (SIG / v / "signatures.npz").exists() and (MASA / f"{v}.npz").exists() and (DINO / f"{v}.npz").exists()]
    groups = {"night": [v for v in sel["night"] if v in vids], "normal": [v for v in sel["normal"] if v in vids]}; groups["all"] = groups["night"] + groups["normal"]
    print(f"[PLAN] видео с полным набором {len(vids)}: ночь {len(groups['night'])}, день {len(groups['normal'])}", flush=True)
    # 1) голова пары на train 130 (как в stitch_eval_v2) и сшивка
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    with ProcessPoolExecutor(max_workers=6, initializer=V._set_dirs, initargs=(150, 5.0, str(DINO), str(MASA))) as pool:
        pairs = list(pool.map(V.video_pairs, [(v, str(SIG), str(CACHE), "val") for v in vids]))
    tasks = [(d["video_id"], str(SIG), str(CACHE), "val", d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), 0.6, str(OUT / "stitched")) for d in pairs]
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(stitch_video, tasks))
    print(f"[STITCH] сшито {len(tasks)} видео; кандидатов {sum(len(d['y']) for d in pairs)}", flush=True)
    # 2) жизни и признаки
    with ProcessPoolExecutor(max_workers=6) as pool:
        L = list(pool.map(video_task, [(v, str(SIG), str(CACHE), "val", "newval", True) for v in vids]))
    pickle.dump(L, open(OUT / "lives.pkl", "wb"))
    seg = {k: int(sum(c["segstats"][k] for c in L)) for k in L[0]["segstats"]}
    print(f"[LIVES] жизней {sum(len(c['label']) for c in L)}; сопоставление рамок {seg}", flush=True)
    # 3) прогнозы зафиксированных моделей
    probs = {var: {} for var in ("nofly", "fly")}
    for var in ("nofly", "fly"):
        m = pickle.load(open(RULE / f"final_{var}.pkl", "rb"))
        for c in L:
            probs[var][c["video_id"]] = m.predict_proba(c[var])[:, 1]
    np.savez_compressed(OUT / "probs.npz", **{f"{var}|{v}": p for var, d in probs.items() for v, p in d.items()})
    # 4) фильтры и HOTA
    res = {}
    for name, (var, thr) in VARIANTS.items():
        tk = [(c["video_id"], str(SIG), str(CACHE), "val", "newval", (np.nonzero(probs[var][c["video_id"]] > thr)[0] if var else []), str(OUT / "filtered" / name.replace("@", "_")), True) for c in L]
        with ProcessPoolExecutor(max_workers=6) as pool:
            res[name] = {r["video_id"]: r for r in pool.map(apply_task, tk)}
    rep = {"videos": vids, "groups": groups, "segstats": seg, "pooled": {}, "removal": {}, "diff": {}, "criterion": {}}
    byv = {c["video_id"]: c for c in L}
    rng = np.random.default_rng(1)
    for g, gv in groups.items():
        if not gv:
            continue
        for name, (var, thr) in VARIANTS.items():
            agg = evaluate_hota(OUT / "filtered" / name.replace("@", "_"), gv, split="val")
            rep["pooled"][f"{name}|{g}"] = {**{k: agg.get(k) for k in KEYS}, "lost": int(sum(res[name][v]["lost"] for v in gv)), "hit": int(sum(res[name][v]["hit"] for v in gv))}
            if var:
                part = [byv[v] for v in gv]; lo = losses(part, probs[var], thr); tp_all = int(sum(int(c["n_tp"].sum()) for c in part))
                kinds = {"false": 0, "mixed": 0, "true": 0, "uncounted": 0}
                for c in part:
                    rm = probs[var][c["video_id"]] > thr
                    kinds["false"] += int((rm & (c["label"] == 1)).sum()); kinds["uncounted"] += int((rm & (c["label"] < 0)).sum())
                    kinds["mixed"] += int((rm & (c["label"] == 0) & (c["n_fp"] > 0)).sum()); kinds["true"] += int((rm & (c["label"] == 0) & (c["n_fp"] == 0)).sum())
                rep["removal"][f"{name}|{g}"] = {"fp_removed": lo["fp_removed"], "tp_removed": lo["tp_removed"], "tp_share": lo["tp_removed"] / max(tp_all, 1), "lost": lo["lost"], "hit": lo["hit"], "lives_removed": kinds}
        for a_, b_ in (("fly@0.7", "nofly@0.85"), ("fly@0.7", "nofly@0.7"), ("fly@0.7", "none"), ("nofly@0.85", "none"), ("nofly@0.7", "none")):
            for met in ("HOTA", "DetA", "AssA"):
                d = np.array([res[a_][v]["metrics"][met] - res[b_][v]["metrics"][met] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                rep["diff"][f"{a_}-{b_}|{g}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        for name in VARIANTS:
            x = rep["pooled"][f"{name}|{g}"]; r = rep["removal"].get(f"{name}|{g}")
            print(f"[{g}] {name:11s}: HOTA {x['HOTA']:.3f} DetA {x['DetA']:.2f} AssA {x['AssA']:.2f} TP {x['CLR_TP']} FP {x['CLR_FP']} FN {x['CLR_FN']}"
                  + (f"; удалено ложных {r['fp_removed']}, верных {r['tp_removed']} ({r['tp_share']:.2%}), жизней {r['lives_removed']}, потеряно {r['lost']}/{r['hit']}" if r else ""), flush=True)
        for k, d in rep["diff"].items():
            if f"|{g}|HOTA" in k:
                print(f"[DIFF] {k}: по видео {d['mean_per_video']:+.3f} [{d['ci95'][0]:+.3f}; {d['ci95'][1]:+.3f}] лучше {d['better']}/{d['n']}", flush=True)
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    n = "night"
    if f"fly@0.7-nofly@0.85|{n}|HOTA" in rep["diff"]:
        d1 = rep["diff"][f"fly@0.7-nofly@0.85|{n}|HOTA"]; d2 = rep["diff"][f"fly@0.7-nofly@0.7|{n}|HOTA"]; d3 = rep["diff"][f"fly@0.7-none|{n}|HOTA"]; rr = rep["removal"][f"fly@0.7|{n}"]
        rep["criterion"] = {"night_fly_minus_nofly_ci_above_0": bool(d1["mean_per_video"] > 0 and d1["ci95"][0] > 0), "night_common_0.7_sign": bool(d2["mean_per_video"] > 0),
                            "night_fly_minus_none_positive": bool(d3["mean_per_video"] > 0), "night_losses_ok": bool(rr["lost"] <= 0.01 * rr["hit"] and rr["tp_share"] <= 0.005)}
    print(f"[CRITERION] {rep['criterion']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
