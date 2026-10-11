"""А2: удаление ложных жизней по временной истории измерений — одинаковая временная голова без FlyVis и с FlyVis
(карточка life_fly_history_v1; протокол зафиксирован 11.10 до расчёта, решение автора).

Данные: жизни train 130 и dev 35 (runs/life_filter_head_v2/lives.pkl: метки v2, прежние признаки, выходы после MASA);
история по шагам — runs/fly_signatures_hist/<id>/history.npz (scripts/build_fly_signatures.py --history; ≤ 32 шагов,
равномерно; жизни и выход трекера сверены с runs/fly_signatures).
Временная голова (одна): по каждому признаку группы — среднее, ст. отклонение, минимум, максимум, наклон по времени
(МНК по номеру шага, нормированному в [0; 1]) и средний модуль приращения между соседними сохранёнными шагами.
  без мухи: прежние 751 + log(1 + число шагов) + статистики входа глаза (721), миниатюры (256), гистограммы (40);
  с мухой: то же + прежние средние мухи (первые / последние 5, 1 292) + статистики выхода мухи (34 × 19 = 646).
Классификатор — HistGradientBoosting конфиг 0 (life_filter_boost.CONFIGS[0], l2 1,0, random_state 0) для обеих голов.
Порог — на train 130 по 5 частям по видео (seed 5), прогнозы вне обучения, правило life_filter_rule_cv (сетка 0,55–0,90
и «ничего не удалять»; допустим, если в каждой части потеря объектов ≤ 1 % и удалённых верных рамок ≤ 0,5 %; выбирается
допустимый с наибольшей средней по видео разностью HOTA с без фильтра при ДИ > 0, иначе «ничего не удалять»).
Итоговые головы — на всём train 130; dev 35 — одна проверка: без фильтра / без мухи / с мухой.
Критерий: (1) с мухой − без мухи по видео > 0, ДИ > 0; (2) с мухой − без фильтра по видео > 0, ДИ > 0;
(3) на dev потеря объектов ≤ 1 %, удалённых верных рамок ≤ 0,5 % (голова с мухой).

Запуск: python scripts/life_fly_history.py   (после конвейера runs/life_fly_history_gpu.cmd)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.life_filter_boost import CONFIGS, LIVES, model  # noqa: E402
from scripts.life_filter_head import apply_task, losses  # noqa: E402

HIST = PROJECT_ROOT / "runs/fly_signatures_hist"
OUT = PROJECT_ROOT / "runs/life_fly_history"
GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, None]
MAX_LOST, MAX_TP = 0.01, 0.005
CFG = CONFIGS[0]
GROUPS_NOFLY = ("hexin", "thumb", "hist")


def stats(X: np.ndarray) -> np.ndarray:
    """(шаги, d) → (6d,): среднее, ст. откл., минимум, максимум, наклон, средний модуль приращения."""
    n = len(X)
    if n == 1:
        z = np.zeros(X.shape[1], np.float32); return np.concatenate([X[0], z, X[0], X[0], z, z])
    t = np.linspace(0.0, 1.0, n, dtype=np.float32); tc = t - t.mean()
    slope = (tc[:, None] * (X - X.mean(0))).sum(0) / (tc ** 2).sum()
    return np.concatenate([X.mean(0), X.std(0), X.min(0), X.max(0), slope, np.abs(np.diff(X, axis=0)).mean(0)])


def video_features(vid):
    """Признаки временной истории для всех жизней видео: (без мухи (n, ·), только муха (n, ·)), float16."""
    z = np.load(HIST / vid / "history.npz"); off = z["offsets"]; ns = z["n_steps"]
    H = {g: z[f"h_{g}"].astype(np.float32).reshape(int(off[-1]), -1) for g in ("fly", *GROUPS_NOFLY)}
    nof, fl = [], []
    for i in range(len(ns)):
        sl = slice(off[i], off[i + 1])
        nof.append(np.concatenate([[np.log1p(ns[i])], *[stats(H[g][sl]) for g in GROUPS_NOFLY]]))
        fl.append(stats(H["fly"][sl]))
    return vid, np.asarray(nof, np.float16), np.asarray(fl, np.float16)


def feats(c, T):
    a, b = T[c["video_id"]]
    nof = np.concatenate([c["nofly"], a.astype(np.float32)], 1)
    return {"nofly": nof, "fly": np.concatenate([nof, c["fly"][:, c["nofly"].shape[1]:], b.astype(np.float32)], 1)}


def boot(d, rng):
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}


def hota(vids, probs, thr, tag):
    tasks = [(c["video_id"], c["sig_dir"], c["root"], c["split"], c["set"], (np.nonzero(probs[c["video_id"]] > thr)[0] if thr is not None else []), str(OUT / "filtered" / tag / c["set"]), True) for c in vids]
    with ProcessPoolExecutor(max_workers=6) as pool:
        return {r["video_id"]: r["metrics"]["HOTA"] for r in pool.map(apply_task, tasks)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    L = pickle.load(open(LIVES, "rb")); tr, dv = L["train"], L["dev"]
    miss = [c["video_id"] for c in tr + dv if not (HIST / c["video_id"] / "history.npz").exists()]
    if miss:
        raise SystemExit(f"нет истории для {len(miss)} видео: {miss[:5]}")
    for c in tr + dv:   # сверка: жизни те же, что в метках
        z = np.load(HIST / c["video_id"] / "signatures.npz"); r = np.load(Path(c["sig_dir"]) / c["video_id"] / "signatures.npz")
        assert len(z["track_id"]) == len(c["label"]) and all(np.array_equal(z[k], r[k]) for k in ("track_id", "start", "end")), c["video_id"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        T = {v: (a, b) for v, a, b in pool.map(video_features, [c["video_id"] for c in tr + dv])}
    print(f"[FEAT] признаки истории готовы: {len(T)} видео | {(time.time() - t0) / 60:.1f} мин", flush=True)
    fold = np.arange(len(tr)) % 5; np.random.default_rng(5).shuffle(fold)
    (OUT / "folds.json").write_text(json.dumps({c["video_id"]: int(f) for c, f in zip(tr, fold)}, indent=1), encoding="utf-8")
    probs = {"nofly": {}, "fly": {}}
    for var in ("nofly", "fly"):
        for f in range(5):
            part = [c for c, k in zip(tr, fold) if k != f]
            y = np.concatenate([c["label"] for c in part]); ok = y >= 0
            m = model(CFG).fit(np.concatenate([feats(c, T)[var][c["label"] >= 0] for c in part]), y[ok])
            for c, k in zip(tr, fold):
                if k == f:
                    probs[var][c["video_id"]] = m.predict_proba(feats(c, T)[var])[:, 1]
            print(f"[OOF] {var}: часть {f} | {(time.time() - t0) / 60:.1f} мин", flush=True)
    rep = {"constraints": {"max_lost": MAX_LOST, "max_tp_share": MAX_TP}, "grid": {}, "chosen": {}, "train_diff": {}, "dev": {}}
    rng = np.random.default_rng(1)
    base_tr = hota(tr, {}, None, "train_none")
    for var in ("nofly", "fly"):
        rep["grid"][var] = {}; best, best_mean = None, None
        for thr in GRID[:-1]:
            per_fold = []
            for f in range(5):
                part = [c for c, k in zip(tr, fold) if k == f]; r = losses(part, probs[var], thr); tp_all = int(sum(int(c["n_tp"].sum()) for c in part))
                per_fold.append({"lost_share": r["lost_share"] or 0.0, "tp_share": r["tp_removed"] / max(tp_all, 1), "fp_removed": r["fp_removed"]})
            ok = all(x["lost_share"] <= MAX_LOST and x["tp_share"] <= MAX_TP for x in per_fold)
            rep["grid"][var][str(thr)] = {"admissible": ok, "folds": per_fold}
            if not ok:
                print(f"[CONSTR] {var} {thr}: недопустим", flush=True); continue
            h = hota(tr, probs[var], thr, f"train_{var}_{thr}")
            d = boot(np.array([h[c["video_id"]] - base_tr[c["video_id"]] for c in tr]), rng); rep["train_diff"][f"{var}@{thr}"] = d
            print(f"[TRAIN] {var} {thr}: − без фильтра {d['mean_per_video']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] лучше {d['better']}/{d['n']}", flush=True)
            if d["ci95"][0] > 0 and (best_mean is None or d["mean_per_video"] > best_mean):
                best, best_mean = thr, d["mean_per_video"]
        rep["chosen"][var] = best
        print(f"[CHOSEN] {var}: порог {best}", flush=True)
    pdev = {}
    for var in ("nofly", "fly"):
        y = np.concatenate([c["label"] for c in tr]); ok = y >= 0
        m = model(CFG).fit(np.concatenate([feats(c, T)[var][c["label"] >= 0] for c in tr]), y[ok])
        pickle.dump(m, open(OUT / f"final_{var}.pkl", "wb"))
        pdev[var] = {c["video_id"]: m.predict_proba(feats(c, T)[var])[:, 1] for c in dv}
    np.savez_compressed(OUT / "probs.npz", **{f"oof_{var}|{v}": p for var, d in probs.items() for v, p in d.items()}, **{f"dev_{var}|{v}": p for var, d in pdev.items() for v, p in d.items()})
    H = {"none": hota(dv, {}, None, "dev_none")}
    for var in ("nofly", "fly"):
        H[var] = hota(dv, pdev[var], rep["chosen"][var], f"dev_{var}") if rep["chosen"][var] is not None else dict(H["none"])
        if rep["chosen"][var] is not None:
            r = losses(dv, pdev[var], rep["chosen"][var]); tp_all = int(sum(int(c["n_tp"].sum()) for c in dv))
            rep["dev"][f"losses_{var}"] = {"lost_share": r["lost_share"], "tp_share": r["tp_removed"] / max(tp_all, 1), "fp_removed": r["fp_removed"], "tp_removed": r["tp_removed"]}
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    night = set(filter_videos_by_subset(load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_train.json")), [c["video_id"] for c in dv], "night"))
    for g, vs in (("all", [c["video_id"] for c in dv]), ("night", [c["video_id"] for c in dv if c["video_id"] in night])):
        rep["dev"][f"mean|{g}"] = {k: float(np.mean([H[k][v] for v in vs])) for k in H}
        for a, b in (("fly", "nofly"), ("fly", "none"), ("nofly", "none")):
            rep["dev"][f"{a}-{b}|{g}"] = boot(np.array([H[a][v] - H[b][v] for v in vs]), rng)
    D = rep["dev"]; lf = D.get("losses_fly")
    rep["criteria"] = {"fly_minus_nofly": D["fly-nofly|all"]["ci95"][0] > 0 and D["fly-nofly|all"]["mean_per_video"] > 0,
                       "fly_minus_none": D["fly-none|all"]["ci95"][0] > 0 and D["fly-none|all"]["mean_per_video"] > 0,
                       "losses_fly": bool(lf is not None and lf["lost_share"] is not None and lf["lost_share"] <= MAX_LOST and lf["tp_share"] <= MAX_TP)}
    rep["criteria"]["all_met"] = all(rep["criteria"].values())
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ("all", "night"):
        print(f"[DEV] {g}: " + ", ".join(f"{k} {v:.3f}" for k, v in D[f'mean|{g}'].items()), flush=True)
        for k in ("fly-nofly", "fly-none", "nofly-none"):
            x = D[f"{k}|{g}"]; print(f"   {k}: {x['mean_per_video']:+.4f} [{x['ci95'][0]:+.4f}; {x['ci95'][1]:+.4f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[LOSSES] {D.get('losses_nofly')} | {lf}", flush=True)
    print(f"[CRITERIA] {rep['criteria']}\n[OK] {(time.time() - t0) / 60:.1f} мин", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
