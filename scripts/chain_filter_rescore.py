"""Сжатая муха в голове удаления цепочки, парная постановка (карточка chain_filter_compact_fly_v1).

chain_filter_rank_v1: сырая сигнатура мухи (1 292 столбца) в голове цепочки ухудшила ранжирование на test/val7. Замена
узла (а), решение автора 08.10 — три сопоставимых варианта, одинаковая голова цепочки (C = 0,1), одинаковые части
проверки dev (runs/chain_filter_rank/folds.json):
  V0 — нынешний фильтр без мухи (748 признаков цепочки);
  V1 — V0 + сжатый выход головы «совпадение рамки начала/конца жизни» БЕЗ мухи;
  V2 — V0 + тот же сжатый выход той же головы С мухой.
Головы жизней (scripts/stitch_realness: конец/начало; без мухи / с мухой) обучаются одинаково на жизнях train 130 с
интерполированными метками; к жизням dev/test/val7 применяются как есть. Свод оценок жизней в цепочку — одинаковый для
V1 и V2: по жизням цепочки среднее, минимум, максимум для r_end и r_start, r_end последней жизни и r_start первой
(8 чисел). Различие V2 − V1 — только муха; V1 − V0 — дополнительное обучение на train 130.
Этап dev (по умолчанию): AP по цепочкам с меткой на предсказаниях вне обучения, бутстрэп по видео (2000); порог с
бюджетом потерь 1 % — удалённые ложные/верные рамки и потери. Критерий перехода (до прогона): AP V2 − AP V1 на dev > 0,
ДИ > 0. Только при выполнении — этап --open: те же величины на test/val7 и HOTA при бюджете 1 %.

Запуск: python scripts/chain_filter_rescore.py [--open]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402
from scripts.chain_filter_head import HEAD, KEYS, apply_filter, chain_task  # noqa: E402
from scripts.chain_filter_rank import GRID, losses  # noqa: E402
from scripts.stitch_realness import X_of, interp_labels, life_features  # noqa: E402

RANK = PROJECT_ROOT / "runs/chain_filter_rank"
OUT = PROJECT_ROOT / "runs/chain_filter_compact"
VARS = ("V0", "V1", "V2")


def agg(c: dict, r_end: np.ndarray, r_start: np.ndarray) -> np.ndarray:
    n = len(c["chains"]); F = np.zeros((n, 8), np.float32)
    lc = c["life_chain"]
    for k in range(n):
        L = np.nonzero(lc == k)[0]
        if not len(L):
            continue
        e, s = r_end[L], r_start[L]
        F[k] = [e.mean(), e.min(), e.max(), s.mean(), s.min(), s.max(), e[np.argmax(c["life_end"][L])], s[np.argmin(c["life_start"][L])]]
    return F


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--open", action="store_true"); a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    splits = ("dev", "test", "val7") if a.open else ("dev",)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    saved = pickle.load(open(RANK / "chains.pkl", "rb")); fold_of = json.loads((RANK / "folds.json").read_text(encoding="utf-8"))
    # 1) головы жизней на train 130 (как в stitch_realness_fly_v1)
    with ProcessPoolExecutor(max_workers=6) as pool:
        tr = [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]
        ftr = list(pool.map(life_features, [str(Path(t[1]) / t[0] / "signatures.npz") for t in tr]))
        for f, lab in zip(ftr, pool.map(interp_labels, tr)):
            f["real"] = lab
    heads = {}
    for side, col in (("end", 0), ("start", 1)):
        y = np.concatenate([f["real"][:, col] for f in ftr]); ok = y >= 0
        for fly in (False, True):
            X = np.concatenate([X_of(f, side, fly) for f in ftr])[ok]
            heads[(side, fly)] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X, y[ok].astype(int))
    del ftr
    print("[HEADS] головы жизней обучены на train 130", flush=True)
    # 2) пересборка цепочек (нужна принадлежность жизней) и сверка с сохранёнными
    data = {}
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        for s in ("train",) + splits:
            data[s] = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    idx = [V.NAMES.index(c) for c in V.HEADS[HEAD]]
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr); del Xtr, ytr, data["train"]
    chains = {}
    for s in splits:
        meta = {v: (str(sr), str(rt), sp) for v, sr, rt, sp in S[s]}
        tasks = [(d["video_id"], *meta[d["video_id"]], s, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1], str(OUT / "repro" / s)) for d in data[s]]
        with ProcessPoolExecutor(max_workers=6) as pool:
            chains[s] = list(pool.map(chain_task, tasks))
        old = {c["video_id"]: c for c in saved["chains"][s]}
        same = all(np.array_equal(c["label"], old[c["video_id"]]["label"]) and np.allclose(c["nofly"], old[c["video_id"]]["nofly"]) for c in chains[s])
        print(f"[CHAINS] {s}: цепочки совпадают с сохранёнными (метки и признаки): {same}", flush=True)
        if not same:
            raise SystemExit("пересборка не совпала с runs/chain_filter_rank/chains.pkl")
        with ProcessPoolExecutor(max_workers=6) as pool:
            lf = list(pool.map(life_features, [str(Path(meta[c["video_id"]][0]) / c["video_id"] / "signatures.npz") for c in chains[s]]))
        for c, f in zip(chains[s], lf):
            for fly, var in ((False, "V1"), (True, "V2")):
                c[var] = np.concatenate([c["nofly"], agg(c, heads[("end", fly)].predict_proba(X_of(f, "end", fly))[:, 1], heads[("start", fly)].predict_proba(X_of(f, "start", fly))[:, 1])], 1)
            c["V0"] = c["nofly"]
    del data
    # 3) голова цепочки: вне обучения на dev, итоговая — на test/val7
    probs = {}
    for var in VARS:
        X = np.concatenate([c[var] for c in chains["dev"]]); y = np.concatenate([c["label"] for c in chains["dev"]])
        g = np.concatenate([np.full(len(c["label"]), fold_of[c["video_id"]]) for c in chains["dev"]]); oof = np.zeros(len(y))
        for f in range(5):
            tr = (g != f) & (y >= 0); te = g == f
            oof[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        off = np.cumsum([0] + [len(c["label"]) for c in chains["dev"]])
        probs[(var, "dev")] = {c["video_id"]: oof[off[i]:off[i + 1]] for i, c in enumerate(chains["dev"])}
        if a.open:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[y >= 0], y[y >= 0])
            for s in ("test", "val7"):
                probs[(var, s)] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in chains[s]}
    np.savez_compressed(OUT / ("probs_open.npz" if a.open else "probs_dev.npz"), **{f"{var}|{s}|{v}": p for (var, s), d in probs.items() for v, p in d.items()})
    rep = {"AP": {}, "AP_diff": {}, "budget": {}, "gate": None}
    rng = np.random.default_rng(1)
    for s in splits:
        lab = {c["video_id"]: c["label"] for c in chains[s]}; vids = list(lab)

        def ap(var, vs):
            y = np.concatenate([lab[v] for v in vs]); p = np.concatenate([probs[(var, s)][v] for v in vs]); k = y >= 0
            return float(average_precision_score(y[k], p[k]))
        for var in VARS:
            rep["AP"][f"{var}|{s}"] = ap(var, vids)
        for x_, y_ in (("V2", "V1"), ("V1", "V0"), ("V2", "V0")):
            bs = []
            for _ in range(2000):
                vs = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
                try:
                    bs.append(ap(x_, vs) - ap(y_, vs))
                except ValueError:
                    pass
            rep["AP_diff"][f"{x_}-{y_}|{s}"] = {"mean": rep["AP"][f"{x_}|{s}"] - rep["AP"][f"{y_}|{s}"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        print(f"[AP] {s}: V0 {rep['AP'][f'V0|{s}']:.4f}, V1 {rep['AP'][f'V1|{s}']:.4f}, V2 {rep['AP'][f'V2|{s}']:.4f}; " + "; ".join(f"{k.split('|')[0]} {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]" for k, d in rep["AP_diff"].items() if k.endswith(f"|{s}")), flush=True)
    for var in VARS:
        grid = {t: losses(chains["dev"], probs[(var, "dev")], t) for t in GRID}
        ok = [t for t in GRID if grid[t]["lost_share"] is not None and grid[t]["lost_share"] <= 0.01]
        thr = max(ok, key=lambda t: (grid[t]["fp_removed"], t)) if ok else None
        rep["budget"][var] = {"threshold": thr}
        for s in splits:
            if thr is not None:
                r = losses(chains[s], probs[(var, s)], thr); rep["budget"][var][s] = {k: v for k, v in r.items() if k != "per_video"}
        if thr is not None:
            r = rep["budget"][var]["dev"]
            print(f"[BUDGET 1 %] {var}: порог {thr}; dev — ложных рамок убрано {r['fp_removed']}, верных {r['tp_removed']}, потеряно {r['lost']}/{r['hit']}, точность удаления {r['precision']:.3f}", flush=True)
    d = rep["AP_diff"]["V2-V1|dev"]; rep["gate"] = bool(d["mean"] > 0 and d["ci95"][0] > 0)
    print(f"[GATE] AP V2 − V1 на dev {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] → критерий перехода {'выполнен' if rep['gate'] else 'НЕ выполнен'}", flush=True)
    if a.open:
        cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
        rep["hota"], rep["hota_diff"] = {}, {}
        for s in ("test", "val7"):
            res = {}
            for var in VARS:
                thr = rep["budget"][var]["threshold"]
                tasks = [(c["video_id"], c["out_dir"], c["split"], c["chains"][probs[(var, s)][c["video_id"]] > thr], str(OUT / "filtered" / var / s)) for c in chains[s]]
                with ProcessPoolExecutor(max_workers=6) as pool:
                    res[var] = {r["video_id"]: r for r in pool.map(apply_filter, tasks)}
                agg_ = evaluate_hota(OUT / "filtered" / var / s, list(res[var]), split=chains[s][0]["split"])
                rep["hota"][f"{var}|{s}"] = {k: agg_.get(k) for k in KEYS}
                rep["hota"][f"{var}|{s}|loss"] = {k: int(sum(x[k] for x in res[var].values())) for k in ("removed_tp_boxes", "lost_gt_objects", "gt_objects_hit")}
            vids = list(res["V0"]); night = [v for v in vids if cond.get(v) == "night"]
            for x_, y_ in (("V2", "V1"), ("V1", "V0"), ("V2", "V0")):
                for g, gv in (("all", vids), ("night", night)):
                    if gv:
                        dd = np.array([res[x_][v]["metrics"]["HOTA"] - res[y_][v]["metrics"]["HOTA"] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                        rep["hota_diff"][f"{x_}-{y_}|{s}|{g}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "n": len(dd)}
            print(f"[HOTA] {s}: " + "; ".join(f"{var} {rep['hota'][f'{var}|{s}']['HOTA']:.3f} (потеряно {rep['hota'][f'{var}|{s}|loss']['lost_gt_objects']})" for var in VARS), flush=True)
    (OUT / ("report_open.json" if a.open else "report_dev.json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
