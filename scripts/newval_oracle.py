"""Запас идеального удаления полностью ложных жизней на 97 новых val-видео (карточка life_only_oracle_newval_v1).

Без GPU: сохранённый выход после MASA (runs/newval_eval/stitched), жизни и метки (runs/newval_eval/lives.pkl), прогнозы
голов (runs/newval_eval/probs.npz). Оракул — удаление всех полностью ложных жизней (метка 1) через
scripts/life_filter_head.apply_task (режим v2); номера и связи не меняются. HOTA, DetA, AssA, потери — ночь / день / все;
разность с выходом без фильтра по видео (бутстрэп 5000). AP ранжирования полностью ложных жизней головами с мухой и без —
описательно, с бутстрэпом по видео разности.

Запуск: python scripts/newval_oracle.py
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

from scripts.life_filter_head import KEYS, apply_task  # noqa: E402

EV = PROJECT_ROOT / "runs/newval_eval"
OUT = PROJECT_ROOT / "runs/newval_oracle"
SIG, CACHE = PROJECT_ROOT / "runs/fly_signatures_newval", PROJECT_ROOT / "runs/newval_cache"


def main() -> int:
    from sklearn.metrics import average_precision_score
    from evaluation.hota_eval import evaluate_hota
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(EV / "lives.pkl", "rb")); rep0 = json.loads((EV / "report.json").read_text(encoding="utf-8"))
    zz = np.load(EV / "probs.npz"); probs = {k: zz[k] for k in zz.files}
    groups = rep0["groups"]
    res = {}
    for name in ("none", "oracle"):
        tk = [(c["video_id"], str(SIG), str(CACHE), "val", "newval", (np.nonzero(c["label"] == 1)[0] if name == "oracle" else []), str(OUT / name), True) for c in L]
        with ProcessPoolExecutor(max_workers=6) as pool:
            res[name] = {r["video_id"]: r for r in pool.map(apply_task, tk)}
    rep = {"pooled": {}, "diff": {}, "AP": {}, "AP_diff": {}, "share_vs_dev": None}
    rng = np.random.default_rng(1)
    byv = {c["video_id"]: c for c in L}
    for g, gv in groups.items():
        for name in ("none", "oracle"):
            agg = evaluate_hota(OUT / name, gv, split="val")
            rep["pooled"][f"{name}|{g}"] = {**{k: agg.get(k) for k in KEYS}, "lost": int(sum(res[name][v]["lost"] for v in gv)), "hit": int(sum(res[name][v]["hit"] for v in gv))}
        for met in ("HOTA", "DetA", "AssA"):
            d = np.array([res["oracle"][v]["metrics"][met] - res["none"][v]["metrics"][met] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"oracle-none|{g}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}

        def ap(var, vs):
            y = np.concatenate([byv[v]["label"] for v in vs]); p = np.concatenate([probs[f"{var}|{v}"] for v in vs]); k = y >= 0
            return float(average_precision_score(y[k], p[k]))
        for var in ("nofly", "fly"):
            rep["AP"][f"{var}|{g}"] = ap(var, gv)
        yy = np.concatenate([byv[v]["label"] for v in gv]); rep["AP"][f"random|{g}"] = float((yy == 1).sum() / max((yy >= 0).sum(), 1))
        bs = []
        for _ in range(2000):
            vs = [gv[i] for i in rng.integers(0, len(gv), len(gv))]
            try:
                bs.append(ap("fly", vs) - ap("nofly", vs))
            except ValueError:
                pass
        rep["AP_diff"][g] = {"mean": rep["AP"][f"fly|{g}"] - rep["AP"][f"nofly|{g}"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        o, n_ = rep["pooled"][f"oracle|{g}"], rep["pooled"][f"none|{g}"]; d = rep["diff"][f"oracle-none|{g}|HOTA"]; a = rep["AP_diff"][g]
        print(f"[{g}] без фильтра {n_['HOTA']:.3f} → оракул {o['HOTA']:.3f} (DetA {n_['DetA']:.2f} → {o['DetA']:.2f}, AssA {n_['AssA']:.2f} → {o['AssA']:.2f}, FP {n_['CLR_FP']} → {o['CLR_FP']}); "
              f"по видео {d['mean_per_video']:+.3f} [{d['ci95'][0]:+.3f}; {d['ci95'][1]:+.3f}] лучше {d['better']}/{d['n']}; "
              f"AP без мухи {rep['AP'][f'nofly|{g}']:.4f}, с мухой {rep['AP'][f'fly|{g}']:.4f} (разность {a['mean']:+.4f} [{a['ci95'][0]:+.4f}; {a['ci95'][1]:+.4f}]), случайное {rep['AP'][f'random|{g}']:.4f}", flush=True)
    n = rep["pooled"]; d = rep["diff"]["oracle-none|night|HOTA"]; gain = n["oracle|night"]["HOTA"] - n["none|night"]["HOTA"]
    rep["criterion"] = {"night_ci_above_0": bool(d["mean_per_video"] > 0 and d["ci95"][0] > 0), "night_gain_ge_0.3": bool(gain >= 0.3), "night_gain": gain}
    print(f"[CRITERION] {rep['criterion']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
