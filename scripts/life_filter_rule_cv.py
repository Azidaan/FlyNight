"""Ограниченный выбор правила удаления жизней по частям train+dev (карточка life_filter_rule_cv_v1, вариант (б) перед (а)).

Решение автора 09.10: зафиксировать правило удаления на существующих данных, затем один раз проверить на новых val-видео.
  Данные: размеченные жизни train 130 + dev 35 (runs/life_filter_head_v2/lives.pkl; выходы — перекрёстная сшивка train и
  сшивка dev). 5 частей по видео (seed 5). Для каждой части обе головы (бустинг, конфигурация 0 из life_filter_boost_v1 —
  одна для обеих) обучаются на остальных частях; прогнозы на части — вне обучения.
  Пороги: {0,55; 0,60; …; 0,90; «ничего не удалять»}. Ограничения (до прогона) — в КАЖДОЙ части: потеря объектов ≤ 1 %
  объектов с попаданиями и удалённых верных рамок ≤ 0,5 % всех верных рамок части (по меткам, правило IoU 0,5).
  Полезность: средняя по видео разность HOTA «фильтр − без фильтра» по всем 165 видео (бутстрэп 5000); выбирается
  допустимый порог с наибольшей средней разностью; если ни один допустимый порог не даёт разности > 0 с ДИ > 0 —
  «ничего не удалять». Процедура одинакова для обеих голов.
  Переход к (а) — для головы выбран порог (не «ничего не удалять»), разность HOTA с без фильтра > 0, ДИ > 0.
  Фиксируется для (а): итоговые модели на train+dev (конфигурация 0) — runs/life_filter_rule_cv/final_<var>.pkl, порог.
  Справочно: «с мухой − без мухи» на выбранных порогах по 165 видео.

Запуск: python scripts/life_filter_rule_cv.py
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
from scripts.life_filter_head import apply_task, losses  # noqa: E402

OUT = PROJECT_ROOT / "runs/life_filter_rule_cv"
GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, None]
MAX_LOST, MAX_TP = 0.01, 0.005
CFG = CONFIGS[0]


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb"))
    vids = L["train"] + L["dev"]
    fold = np.arange(len(vids)) % 5; np.random.default_rng(5).shuffle(fold)
    (OUT / "folds.json").write_text(json.dumps({c["video_id"]: int(f) for c, f in zip(vids, fold)}, indent=1), encoding="utf-8")
    probs = {"nofly": {}, "fly": {}}
    for var in ("nofly", "fly"):
        for f in range(5):
            tr = [c for c, k in zip(vids, fold) if k != f]
            y = np.concatenate([c["label"] for c in tr]); ok = y >= 0
            m = model(CFG).fit(np.concatenate([c[var] for c in tr])[ok], y[ok])
            for c, k in zip(vids, fold):
                if k == f:
                    probs[var][c["video_id"]] = m.predict_proba(c[var])[:, 1]
            print(f"[OOF] {var}: часть {f} готова", flush=True)
        y = np.concatenate([c["label"] for c in vids]); ok = y >= 0
        pickle.dump(model(CFG).fit(np.concatenate([c[var] for c in vids])[ok], y[ok]), open(OUT / f"final_{var}.pkl", "wb"))
    np.savez_compressed(OUT / "probs_oof.npz", **{f"{var}|{v}": p for var, d in probs.items() for v, p in d.items()})
    rep = {"constraints": {"max_lost": MAX_LOST, "max_tp_share": MAX_TP}, "grid": {}, "chosen": {}, "diff": {}, "transition": {}}
    # 1) ограничения по частям (по меткам, без HOTA)
    for var in ("nofly", "fly"):
        rep["grid"][var] = {}
        for thr in GRID:
            per_fold = []
            for f in range(5):
                part = [c for c, k in zip(vids, fold) if k == f]
                if thr is None:
                    per_fold.append({"lost_share": 0.0, "tp_share": 0.0, "fp_removed": 0}); continue
                r = losses(part, probs[var], thr); tp_all = int(sum(int(c["n_tp"].sum()) for c in part))
                per_fold.append({"lost_share": r["lost_share"] or 0.0, "tp_share": r["tp_removed"] / max(tp_all, 1), "fp_removed": r["fp_removed"], "tp_removed": r["tp_removed"]})
            ok = all(x["lost_share"] <= MAX_LOST and x["tp_share"] <= MAX_TP for x in per_fold)
            rep["grid"][var][str(thr)] = {"admissible": ok, "folds": per_fold, "fp_removed": int(sum(x["fp_removed"] for x in per_fold)), "tp_removed": int(sum(x.get("tp_removed", 0) for x in per_fold))}
            print(f"[CONSTR] {var} порог {thr}: допустим {ok}; по частям потери объектов {[round(x['lost_share'] * 100, 2) for x in per_fold]} %, верных рамок {[round(x['tp_share'] * 100, 2) for x in per_fold]} %; ложных убрано {rep['grid'][var][str(thr)]['fp_removed']}", flush=True)
    # 2) HOTA по видео — только для допустимых порогов
    def hota(var, thr):
        tag = f"{var}_{thr}"
        tasks = [(c["video_id"], c["sig_dir"], c["root"], c["split"], c["set"], (np.nonzero(probs[var][c["video_id"]] > thr)[0] if thr is not None else []), str(OUT / "filtered" / tag / c["set"]), True) for c in vids]
        with ProcessPoolExecutor(max_workers=6) as pool:
            return {r["video_id"]: r["metrics"]["HOTA"] for r in pool.map(apply_task, tasks)}
    base = hota("none", None)
    rng = np.random.default_rng(1); per_thr = {}
    for var in ("nofly", "fly"):
        best, best_mean = "None", None
        for thr in GRID[:-1]:
            if not rep["grid"][var][str(thr)]["admissible"]:
                continue
            h = hota(var, thr); per_thr[(var, thr)] = h
            d = np.array([h[c["video_id"]] - base[c["video_id"]] for c in vids]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{var}@{thr}-none"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
            r = rep["diff"][f"{var}@{thr}-none"]
            print(f"[HOTA] {var} порог {thr}: − без фильтра по видео {r['mean_per_video']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}] лучше {r['better']}/{r['n']}", flush=True)
            if r["ci95"][0] > 0 and (best_mean is None or r["mean_per_video"] > best_mean):
                best, best_mean = str(thr), r["mean_per_video"]
        rep["chosen"][var] = best
        rep["transition"][var] = best != "None"
        print(f"[CHOSEN] {var}: порог {best}", flush=True)
    if rep["chosen"]["nofly"] != "None" and rep["chosen"]["fly"] != "None":
        a = per_thr[("fly", float(rep["chosen"]["fly"]))]; b = per_thr[("nofly", float(rep["chosen"]["nofly"]))]
        d = np.array([a[c["video_id"]] - b[c["video_id"]] for c in vids]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        rep["diff"]["fly-nofly@chosen"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        r = rep["diff"]["fly-nofly@chosen"]
        print(f"[FLY] с мухой − без мухи на выбранных порогах (справочно): {r['mean_per_video']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}] лучше {r['better']}/{r['n']}", flush=True)
    print(f"[TRANSITION] {rep['transition']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
