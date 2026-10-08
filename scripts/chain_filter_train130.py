"""Голова удаления цепочки на цепочках train 130 (карточка chain_filter_train130_v1) — замена (б): больше данных.

chain_filter_rank_v1 / chain_filter_compact_fly_v1: голова цепочки училась на dev 35 (~4 500 цепочек с меткой), сырая
муха (1 292 столбца) ухудшила ранжирование на test/val7 — одна из версий — мало данных. Здесь голова учится на цепочках
train 130. Утечка: голова пары (база + MASA) обучена на train 130 — сшивка train ею была бы лучше, чем на новых видео.
Поэтому перекрёстная подгонка: train делится пополам по видео (seed 11); каждая половина сшивается головой пары,
обученной на другой половине (жадно, порог 0,6, окно 150; 5). dev/test/val7 — как раньше (голова пары на всём train).
Признаки цепочки, метки (правило output_filter_oracle), голова (логистическая регрессия C = 0,1) — те же, что в
chain_filter_rank_v1; варианты без мухи (748) и с мухой (2 040). dev — чистая проверка: AP, порог при бюджете потерь 1 %
(наибольшее число удалённых ложных рамок при потерях ≤ 1 %). Критерий перехода (до прогона): AP с мухой − без мухи на dev
> 0, ДИ по видео > 0; только тогда --open: AP и HOTA при бюджете 1 % на test/val7.
Справочно: AP без мухи (train 130) против прежнего AP без мухи на dev вне обучения (0,534, chain_filter_rank_v1).

Запуск: python scripts/chain_filter_train130.py [--open]
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

RANK = PROJECT_ROOT / "runs/chain_filter_rank"
OUT = PROJECT_ROOT / "runs/chain_filter_train130"
VARS = ("nofly", "fly")


def pair_head(ds):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = [V.NAMES.index(c) for c in V.HEADS[HEAD]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in ds]); y = np.concatenate([d["y"][d["y"] >= 0] for d in ds])
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y), idx


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--open", action="store_true"); a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    meta = {s: {v: (str(sr), str(rt), sp) for v, sr, rt, sp in S[s]} for s in S}
    cache = OUT / "train_chains.pkl"
    if cache.exists():
        train_chains, halves = pickle.load(open(cache, "rb"))
    else:
        with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
            tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
        order = np.random.default_rng(11).permutation(len(tr)); halves = {tr[i]["video_id"]: int(k % 2) for k, i in enumerate(order)}
        train_chains = []
        for h in (0, 1):
            m, idx = pair_head([d for d in tr if halves[d["video_id"]] != h])
            tasks = [(d["video_id"], *meta["train"][d["video_id"]], "train", d["pairs"], m.predict_proba(d["X"][:, idx])[:, 1], str(OUT / "repro" / "train"))
                     for d in tr if halves[d["video_id"]] == h]
            with ProcessPoolExecutor(max_workers=6) as pool:
                train_chains += list(pool.map(chain_task, tasks))
            print(f"[TRAIN] половина {h}: сшито {len(tasks)} видео головой пары с другой половины", flush=True)
        del tr
        pickle.dump((train_chains, halves), open(cache, "wb"))
    (OUT / "halves.json").write_text(json.dumps(halves, indent=1), encoding="utf-8")
    lab_tr = np.concatenate([c["label"] for c in train_chains])
    print(f"[TRAIN] цепочек {len(lab_tr)}: ложных {(lab_tr == 1).sum()}, оставить {(lab_tr == 0).sum()}, без учитываемых рамок {(lab_tr < 0).sum()}", flush=True)
    saved = pickle.load(open(RANK / "chains.pkl", "rb"))["chains"]   # dev/test/val7 — те же цепочки, что в chain_filter_rank_v1
    splits = ("dev", "test", "val7") if a.open else ("dev",)
    probs, rep = {}, {"counts_train": {"chains": int(len(lab_tr)), "false": int((lab_tr == 1).sum()), "keep": int((lab_tr == 0).sum())}, "AP": {}, "AP_diff": {}, "budget": {}, "gate": None}
    for var in VARS:
        X = np.concatenate([c[var] for c in train_chains]); y = lab_tr
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[y >= 0], y[y >= 0])
        for s in splits:
            probs[(var, s)] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in saved[s]}
    np.savez_compressed(OUT / ("probs_open.npz" if a.open else "probs_dev.npz"), **{f"{var}|{s}|{v}": p for (var, s), d in probs.items() for v, p in d.items()})
    rng = np.random.default_rng(1)
    for s in splits:
        lab = {c["video_id"]: c["label"] for c in saved[s]}; vids = list(lab)

        def ap(var, vs):
            yy = np.concatenate([lab[v] for v in vs]); pp = np.concatenate([probs[(var, s)][v] for v in vs]); k = yy >= 0
            return float(average_precision_score(yy[k], pp[k]))
        for var in VARS:
            rep["AP"][f"{var}|{s}"] = ap(var, vids)
        bs = []
        for _ in range(2000):
            vs = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
            try:
                bs.append(ap("fly", vs) - ap("nofly", vs))
            except ValueError:
                pass
        rep["AP_diff"][s] = {"mean": rep["AP"][f"fly|{s}"] - rep["AP"][f"nofly|{s}"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        d = rep["AP_diff"][s]
        print(f"[AP] {s}: без мухи {rep['AP'][f'nofly|{s}']:.4f}, с мухой {rep['AP'][f'fly|{s}']:.4f}, разность {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]", flush=True)
    prev = json.loads((RANK / "report_rank.json").read_text(encoding="utf-8"))["AP"]
    rep["AP_prev_dev_oof"] = {"nofly": prev["nofly|dev"], "fly": prev["fly|dev"]}
    print(f"[AP] справочно, dev: без мухи train 130 {rep['AP']['nofly|dev']:.4f} против прежнего (dev вне обучения) {prev['nofly|dev']:.4f}; с мухой {rep['AP']['fly|dev']:.4f} против {prev['fly|dev']:.4f}", flush=True)
    for var in VARS:
        grid = {t: losses(saved["dev"], probs[(var, "dev")], t) for t in GRID}
        ok = [t for t in GRID if grid[t]["lost_share"] is not None and grid[t]["lost_share"] <= 0.01]
        thr = max(ok, key=lambda t: (grid[t]["fp_removed"], t)) if ok else None
        rep["budget"][var] = {"threshold": thr}
        if thr is not None:
            for s in splits:
                r = losses(saved[s], probs[(var, s)], thr); rep["budget"][var][s] = {k: v for k, v in r.items() if k != "per_video"}
            r = rep["budget"][var]["dev"]
            print(f"[BUDGET 1 %] {var}: порог {thr}; dev — ложных рамок убрано {r['fp_removed']}, верных {r['tp_removed']}, потеряно {r['lost']}/{r['hit']}, точность удаления {r['precision']:.3f}", flush=True)
        else:
            print(f"[BUDGET 1 %] {var}: допустимого порога на dev нет", flush=True)
    d = rep["AP_diff"]["dev"]; rep["gate"] = bool(d["mean"] > 0 and d["ci95"][0] > 0)
    print(f"[GATE] AP с мухой − без мухи на dev {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] → критерий перехода {'выполнен' if rep['gate'] else 'НЕ выполнен'}", flush=True)
    if a.open:
        cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
        rep["hota"], rep["hota_diff"] = {}, {}
        for s in ("test", "val7"):
            res = {}
            for var in ("none",) + VARS:
                thr = rep["budget"].get(var, {}).get("threshold", 2.0) if var != "none" else 2.0
                tasks = [(c["video_id"], c["out_dir"], c["split"], c["chains"][(probs[(var, s)][c["video_id"]] if var != "none" else np.zeros(len(c["chains"]))) > thr], str(OUT / "filtered" / var / s)) for c in saved[s]]
                with ProcessPoolExecutor(max_workers=6) as pool:
                    res[var] = {r["video_id"]: r for r in pool.map(apply_filter, tasks)}
                agg_ = evaluate_hota(OUT / "filtered" / var / s, list(res[var]), split=saved[s][0]["split"])
                rep["hota"][f"{var}|{s}"] = {k: agg_.get(k) for k in KEYS}
                rep["hota"][f"{var}|{s}|loss"] = {k: int(sum(x[k] for x in res[var].values())) for k in ("removed_tp_boxes", "lost_gt_objects", "gt_objects_hit")}
            vids = list(res["none"]); night = [v for v in vids if cond.get(v) == "night"]
            for x_, y_ in (("fly", "nofly"), ("nofly", "none"), ("fly", "none")):
                for g, gv in (("all", vids), ("night", night)):
                    if gv:
                        dd = np.array([res[x_][v]["metrics"]["HOTA"] - res[y_][v]["metrics"]["HOTA"] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                        rep["hota_diff"][f"{x_}-{y_}|{s}|{g}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "n": len(dd)}
            print(f"[HOTA] {s}: " + "; ".join(f"{var} {rep['hota'][f'{var}|{s}']['HOTA']:.3f} (потеряно {rep['hota'][f'{var}|{s}|loss']['lost_gt_objects']})" for var in ("none",) + VARS), flush=True)
    (OUT / ("report_open.json" if a.open else "report_dev.json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
