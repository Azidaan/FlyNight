"""Правило выбора порога фильтра жизней с учётом потери верных рамок (карточка life_filter_rule_v1, шаг 4.2).

life_removal_harm_diag_v1: вред фильтра — удаление целиком настоящих коротких жизней, равномерно по группам; правило
«бюджет потерь 1 % объектов» не учитывает удалённые верные рамки (на val7 с мухой верных удалено больше, чем ложных).
Здесь — те же головы (бустинг без мухи / с мухой, runs/life_filter_boost_hota/probs.npz) и метки (lives.pkl v2):
  правило H (основное): на dev — порог из сетки {0,50; 0,55; …; 0,95; «ничего не удалять»} с наибольшим общим HOTA при
    потере объектов ≤ 1 % (HOTA видит и убранные ложные, и потерянные верные рамки);
  правило P (справочное, по меткам): наибольшее «удалённые ложные рамки − удалённые верные / DetA» (DetA исходного выхода dev).
Обе головы — одно правило. test и val7 — диагностика (уже прочитаны): HOTA, DetA, AssA, FP/FN, потери, верных на ложную,
парные разности по видео (бутстрэп 5000), ночь. Окончательная проверка — на новых val-видео (шаг 4.3).
Переход к 4.3 (до прогона): на test и val7 правило H для каждой головы — HOTA − без фильтра по видео ≥ 0 (знак) и верных
удалённых рамок на ложную меньше, чем при бюджете 1 %; если H выбирает «ничего не удалять», перехода нет.

Запуск: python scripts/life_filter_rule.py
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

from scripts.life_filter_head import KEYS, apply_task, losses  # noqa: E402

LIVES = PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl"
PROBS = PROJECT_ROOT / "runs/life_filter_boost_hota/probs.npz"
BASE = PROJECT_ROOT / "runs/life_filter_boost_hota"
OUT = PROJECT_ROOT / "runs/life_filter_rule"
GRID = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)] + [None]   # None — ничего не удалять


def run(L, z, s, var, thr, tag):
    tasks = [(c["video_id"], c["sig_dir"], c["root"], c["split"], s, (np.nonzero(z[f"{var}|{s}|{c['video_id']}"] > thr)[0] if thr is not None else []), str(OUT / "filtered" / tag / s), True) for c in L[s]]
    with ProcessPoolExecutor(max_workers=6) as pool:
        return {r["video_id"]: r for r in pool.map(apply_task, tasks)}


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb")); zz = np.load(PROBS); z = {k: zz[k] for k in zz.files}
    old = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    rep = {"grid": {}, "rule_H": {}, "rule_P": {}, "pooled": {}, "diff": {}, "ratio": {}, "transition": {}}
    det0 = old["pooled"]["none|dev|all"]["DetA"] / 100
    for var in ("nofly", "fly"):
        grid = {}
        for thr in GRID:
            tag = f"{var}_{thr if thr is not None else 'none'}"
            per = run(L, z, "dev", var, thr, tag)
            agg = evaluate_hota(OUT / "filtered" / tag / "dev", list(per), split=L["dev"][0]["split"])
            lab = losses(L["dev"], {c["video_id"]: (z[f"{var}|dev|{c['video_id']}"] if thr is not None else np.zeros(len(c["label"]))) for c in L["dev"]}, thr if thr is not None else 2.0)
            grid[str(thr)] = {"HOTA": agg["HOTA"], "DetA": agg["DetA"], "AssA": agg["AssA"], "lost_share": lab["lost_share"] or 0.0, "fp_removed": lab["fp_removed"], "tp_removed": lab["tp_removed"],
                              "proxy": lab["fp_removed"] - lab["tp_removed"] / det0}
            print(f"[GRID] {var} порог {thr}: HOTA dev {agg['HOTA']:.3f}, ложных убрано {lab['fp_removed']}, верных {lab['tp_removed']}, потери {grid[str(thr)]['lost_share']:.2%}, P {grid[str(thr)]['proxy']:.1f}", flush=True)
        rep["grid"][var] = grid
        ok = [k for k, g in grid.items() if g["lost_share"] <= 0.01]
        rep["rule_H"][var] = max(ok, key=lambda k: grid[k]["HOTA"])
        rep["rule_P"][var] = max(ok, key=lambda k: grid[k]["proxy"])
        print(f"[RULE] {var}: H → {rep['rule_H'][var]}, P → {rep['rule_P'][var]}, прежний бюджет 1 % → {old['thresholds'][var]}", flush=True)
    rng = np.random.default_rng(1)
    for s in ("test", "val7"):
        split = L[s][0]["split"]; res = {}
        res["none"] = run(L, z, s, "nofly", None, "none")
        for var in ("nofly", "fly"):
            for rule in ("H", "P"):
                k = rep[f"rule_{rule}"][var]; thr = None if k == "None" else float(k)
                res[f"{var}@{rule}"] = run(L, z, s, var, thr, f"{var}_{k}")
                lab = losses(L[s], {c["video_id"]: (z[f"{var}|{s}|{c['video_id']}"] if thr is not None else np.zeros(len(c["label"]))) for c in L[s]}, thr if thr is not None else 2.0)
                rep["ratio"][f"{var}@{rule}|{s}"] = {"fp_removed": lab["fp_removed"], "tp_removed": lab["tp_removed"], "tp_per_fp": lab["tp_removed"] / max(lab["fp_removed"], 1)}
            ob = losses(L[s], {c["video_id"]: z[f"{var}|{s}|{c['video_id']}"] for c in L[s]}, old["thresholds"][var])
            rep["ratio"][f"{var}@budget1|{s}"] = {"fp_removed": ob["fp_removed"], "tp_removed": ob["tp_removed"], "tp_per_fp": ob["tp_removed"] / max(ob["fp_removed"], 1)}
        vids = list(res["none"])
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        groups = {"all": vids, "night": list(filter_videos_by_subset(attrs, vids, "night"))}
        for key, per in res.items():
            k_dir = "none" if key == "none" else f"{key.split('@')[0]}_{rep['rule_' + key.split('@')[1]][key.split('@')[0]]}"
            for g, gv in groups.items():
                if gv:
                    agg = evaluate_hota(OUT / "filtered" / k_dir / s, gv, split=split)
                    rep["pooled"][f"{key}|{s}|{g}"] = {**{m: agg.get(m) for m in KEYS}, "lost": int(sum(per[v]["lost"] for v in gv)), "hit": int(sum(per[v]["hit"] for v in gv))}
        for a_, b_ in [(f"{v}@{r}", "none") for v in ("nofly", "fly") for r in ("H", "P")] + [("fly@H", "nofly@H"), ("fly@P", "nofly@P")]:
            for g, gv in groups.items():
                if gv:
                    d = np.array([res[a_][v]["metrics"]["HOTA"] - res[b_][v]["metrics"]["HOTA"] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    rep["diff"][f"{a_}-{b_}|{s}|{g}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        for key in res:
            x = rep["pooled"][f"{key}|{s}|all"]; ra = rep["ratio"].get(f"{key}|{s}")
            print(f"[{s}] {key:9s}: HOTA {x['HOTA']:.3f} DetA {x['DetA']:.2f} AssA {x['AssA']:.2f} FP {x['CLR_FP']} FN {x['CLR_FN']}; потеряно {x['lost']}/{x['hit']}" + (f"; верных на ложную {ra['tp_per_fp']:.2f}" if ra else ""), flush=True)
        for k, d in rep["diff"].items():
            if f"|{s}|all" in k:
                print(f"[DIFF] {k}: {d['mean_per_video']:+.3f} [{d['ci95'][0]:+.3f}; {d['ci95'][1]:+.3f}] лучше {d['better']}/{d['n']}", flush=True)
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for var in ("nofly", "fly"):
        rep["transition"][var] = rep["rule_H"][var] != "None" and all(rep["diff"][f"{var}@H-none|{s}|all"]["mean_per_video"] >= 0 and rep["ratio"][f"{var}@H|{s}"]["tp_per_fp"] < rep["ratio"][f"{var}@budget1|{s}"]["tp_per_fp"] for s in ("test", "val7"))
    print(f"[TRANSITION] {rep['transition']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
