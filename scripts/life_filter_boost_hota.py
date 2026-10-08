"""HOTA головы отдельной жизни (бустинг) при бюджете потерь 1 % — без мухи и с мухой (карточка life_filter_boost_hota_v1).

life_filter_boost_v1: на dev бустинг с мухой ранжирует ложные жизни лучше бустинга без мухи (+0,0148 AP, ДИ выше 0).
Здесь — тот же протокол до HOTA: модели переобучаются с выбранными конфигурациями на train 130 (детерминированно; сверка —
AP на dev совпадает с report_dev.json), пороги — из бюджета 1 % на dev (runs/life_filter_boost/report_dev.json), удаление —
записи жизней с вероятностью выше порога из исходного выхода после MASA (номера и связи не меняются;
scripts/life_filter_head.apply_task, режим v2). Наборы: dev (порог выбран на нём — справочно), test 35, val7.
Варианты: без фильтра; бустинг без мухи; бустинг с мухой; оракул того же действия (удаление всех полностью ложных жизней) —
знаменатель доли потолка. Отчёт: HOTA, DetA, AssA, CLR_TP/FP/FN общие; потерянные найденные объекты; парные разности по
видео (бутстрэп 5000), ночь (datasets.subset_filter).
Критерий (до прогона): test — HOTA «с мухой − без мухи» по видео > 0, ДИ > 0; val7 — тот же знак; потери с мухой на test
≤ 1 % объектов с попаданиями.

Запуск: python scripts/life_filter_boost_hota.py
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

from scripts.life_filter_boost import CONFIGS, LIVES, model  # noqa: E402
from scripts.life_filter_head import KEYS, apply_task  # noqa: E402

BOOST = PROJECT_ROOT / "runs/life_filter_boost"
OUT = PROJECT_ROOT / "runs/life_filter_boost_hota"


def main() -> int:
    from sklearn.metrics import average_precision_score
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb")); rb = json.loads((BOOST / "report_dev.json").read_text(encoding="utf-8"))
    tr = L["train"]; ytr = np.concatenate([c["label"] for c in tr]); ok = ytr >= 0
    probs, rep = {}, {"thresholds": {v: rb["budget"][v]["threshold"] for v in ("nofly", "fly")}, "AP_check": {}, "pooled": {}, "diff": {}, "share": {}}
    for var in ("nofly", "fly"):
        m = model(CONFIGS[rb["chosen"][var]]).fit(np.concatenate([c[var] for c in tr])[ok], ytr[ok])
        for s in ("dev", "test", "val7"):
            probs[(var, s)] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in L[s]}
        y = np.concatenate([c["label"] for c in L["dev"]]); p = np.concatenate([probs[(var, "dev")][c["video_id"]] for c in L["dev"]])
        rep["AP_check"][var] = {"now": float(average_precision_score(y[y >= 0], p[y >= 0])), "report": rb["AP"][var]}
        print(f"[CHECK] {var}: AP dev сейчас {rep['AP_check'][var]['now']:.4f}, в отчёте {rb['AP'][var]:.4f}", flush=True)
    np.savez_compressed(OUT / "probs.npz", **{f"{var}|{s}|{v}": p for (var, s), d in probs.items() for v, p in d.items()})
    rng = np.random.default_rng(1)
    for s in ("dev", "test", "val7"):
        split = L[s][0]["split"]; res = {}
        drops = {"none": lambda c: [], "oracle": lambda c: np.nonzero(c["label"] == 1)[0]}
        for var in ("nofly", "fly"):
            thr = rep["thresholds"][var]
            drops[var] = (lambda c, var=var, thr=thr: np.nonzero(probs[(var, s)][c["video_id"]] > thr)[0])
        for var, fn in drops.items():
            tasks = [(c["video_id"], c["sig_dir"], c["root"], c["split"], s, fn(c), str(OUT / "filtered" / var / s), True) for c in L[s]]
            with ProcessPoolExecutor(max_workers=6) as pool:
                res[var] = {r["video_id"]: r for r in pool.map(apply_task, tasks)}
        vids = list(res["none"])
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        groups = {"all": vids, "night": list(filter_videos_by_subset(attrs, vids, "night"))}
        for g, gv in groups.items():
            if not gv:
                continue
            for var in res:
                agg = evaluate_hota(OUT / "filtered" / var / s, gv, split=split)
                rep["pooled"][f"{var}|{s}|{g}"] = {**{k: agg.get(k) for k in KEYS}, "lost": int(sum(res[var][v]["lost"] for v in gv)), "hit": int(sum(res[var][v]["hit"] for v in gv))}
            for a_, b_ in (("fly", "nofly"), ("nofly", "none"), ("fly", "none"), ("oracle", "none")):
                for met in ("HOTA", "DetA", "AssA"):
                    d = np.array([res[a_][v]["metrics"][met] - res[b_][v]["metrics"][met] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    rep["diff"][f"{a_}-{b_}|{s}|{g}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        H = lambda v: rep["pooled"][f"{v}|{s}|all"]["HOTA"]
        den = H("oracle") - H("none")
        for var in ("nofly", "fly"):
            rep["share"][f"{var}|{s}"] = (H(var) - H("none")) / den if den > 0 else None
        for var in ("none", "nofly", "fly", "oracle"):
            x = rep["pooled"][f"{var}|{s}|all"]
            print(f"[HOTA] {s} {var:7s}: HOTA {x['HOTA']:.3f} DetA {x['DetA']:.2f} AssA {x['AssA']:.2f} TP {x['CLR_TP']} FP {x['CLR_FP']} FN {x['CLR_FN']}; потеряно {x['lost']}/{x['hit']}", flush=True)
        for k in (f"fly-nofly|{s}|all|HOTA", f"nofly-none|{s}|all|HOTA", f"fly-none|{s}|all|HOTA", f"fly-nofly|{s}|night|HOTA"):
            if k in rep["diff"]:
                r = rep["diff"][k]; print(f"[DIFF] {k}: по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
        print(f"[SHARE] {s}: доля потолка того же действия — без мухи {rep['share'][f'nofly|{s}']}, с мухой {rep['share'][f'fly|{s}']}", flush=True)
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    d = rep["diff"]["fly-nofly|test|all|HOTA"]; dv = rep["diff"]["fly-nofly|val7|all|HOTA"]; lf = rep["pooled"]["fly|test|all"]
    rep["criterion"] = {"test_ci_above_0": bool(d["mean_per_video"] > 0 and d["ci95"][0] > 0), "val7_sign": bool(dv["mean_per_video"] > 0), "test_loss_le_1pct": bool(lf["lost"] <= 0.01 * lf["hit"])}
    print(f"[CRITERION] {rep['criterion']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
