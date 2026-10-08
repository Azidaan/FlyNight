"""Ранжирование и порог с бюджетом потерь для головы «удалить итоговую цепочку» (карточка chain_filter_rank_v1).

Продолжение chain_filter_fly_v1: там пороги вариантов встали в разные точки плоской кривой dev, и сравнение смешало
различающую способность головы и выбор рабочей точки. Здесь (решение автора 08.10), тот же протокол и те же признаки:
  этап rank (без оценщика HOTA):
    * пересборка выхода после MASA и цепочек (scripts/chain_filter_head.chain_task), признаки, метки, верные/ложные рамки
      цепочек и покрываемые объекты разметки — сохраняются на диск (runs/chain_filter_rank/chains.pkl);
    * части проверки dev (5, по видео; та же раскладка, что в chain_filter_fly_v1) — сохраняются; предсказания вне
      обучения на dev и предсказания итоговых моделей (обучены на всём dev) на test/val7 — probs.npz;
    * AP головы по цепочкам с меткой (0/1), бутстрэп по видео (2000) для «с мухой − без мухи»;
    * по сетке порогов: удалено учитываемых цепочек, точность удаления (доля полностью ложных среди удалённых
      учитываемых), удалено ложных и верных рамок, потеряно объектов разметки (все их верные рамки — в удалённых
      цепочках) — без оценщика, по правилу output_filter_oracle;
    * бюджет потерь B ∈ {0,5 %; 1 %; 2 %} объектов разметки с попаданиями: на dev (вне обучения) — порог с наибольшим
      числом удалённых ложных рамок при потерях ≤ B; тот же порог — на test/val7.
  этап hota (--hota): полный пересчёт HOTA (TrackEval) только для выбранных порогов, без фильтра, обоих вариантов;
    парные разности по видео (бутстрэп 5000), потери, ночь/день.
Кривая по доле всех удалённых цепочек — только вспомогательная (72 % цепочек test без учитываемых рамок).

Запуск: python scripts/chain_filter_rank.py            (этап rank)
        python scripts/chain_filter_rank.py --hota     (этап hota)
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

OUT = PROJECT_ROOT / "runs/chain_filter_rank"
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
BUDGETS = [0.005, 0.01, 0.02]
VARS = ("nofly", "fly")


def losses(ch: list, probs: dict, thr: float) -> dict:
    """Потери и выигрыш удаления цепочек с p > thr по правилу оракула (без оценщика)."""
    out = {"removed_counted": 0, "removed_false": 0, "removed_uncounted": 0, "fp_removed": 0, "tp_removed": 0, "lost": 0, "hit": 0, "per_video": {}}
    for c in ch:
        p = probs[c["video_id"]]; rm = p > thr; lab = c["label"]
        tot, gone = {}, {}
        for ci, g, k in c["tp_map"]:
            tot[g] = tot.get(g, 0) + k
            if rm[ci]:
                gone[g] = gone.get(g, 0) + k
        lost = sum(1 for g in tot if gone.get(g, 0) == tot[g])
        v = {"removed_counted": int((rm & (lab >= 0)).sum()), "removed_false": int((rm & (lab == 1)).sum()), "removed_uncounted": int((rm & (lab < 0)).sum()),
             "fp_removed": int(c["n_fp"][rm].sum()), "tp_removed": int(c["n_tp"][rm].sum()), "lost": lost, "hit": len(tot)}
        out["per_video"][c["video_id"]] = v
        for k in v:
            out[k] += v[k]
    out["precision"] = out["removed_false"] / out["removed_counted"] if out["removed_counted"] else None
    out["lost_share"] = out["lost"] / out["hit"] if out["hit"] else None
    return out


FIXED = False   # --fixed (проверка влияния ошибки 09.10): признаки цепочки без повторов строк номера; выход — runs/chain_filter_rank_fixed


def stage_rank() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    data = {}
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        for s in ("train", "dev", "test", "val7"):
            data[s] = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    idx = [V.NAMES.index(c) for c in V.HEADS[HEAD]]
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr)
    del Xtr, ytr, data["train"]
    chains = {}
    for s in ("dev", "test", "val7"):
        meta = {v: (str(sr), str(rt), sp) for v, sr, rt, sp in S[s]}
        tasks = [(d["video_id"], *meta[d["video_id"]], s, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1], str(OUT / "repro" / s), FIXED) for d in data[s]]
        with ProcessPoolExecutor(max_workers=6) as pool:
            chains[s] = list(pool.map(chain_task, tasks))
        print(f"[CHAINS] {s}: видео {len(chains[s])}, цепочек {sum(len(c['label']) for c in chains[s])}", flush=True)
    del data
    folds = np.arange(len(chains["dev"])) % 5; np.random.default_rng(7).shuffle(folds)   # как в chain_filter_fly_v1
    fold_of = {c["video_id"]: int(f) for c, f in zip(chains["dev"], folds)}
    with open(OUT / "chains.pkl", "wb") as fh:
        pickle.dump({"chains": chains, "fold_of": fold_of}, fh)
    probs, models_info = {}, {}
    for var in VARS:
        X = np.concatenate([c[var] for c in chains["dev"]]); y = np.concatenate([c["label"] for c in chains["dev"]])
        g = np.concatenate([np.full(len(c["label"]), fold_of[c["video_id"]]) for c in chains["dev"]]); oof = np.zeros(len(y))
        for f in range(5):
            tr = (g != f) & (y >= 0); te = g == f
            oof[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        off = np.cumsum([0] + [len(c["label"]) for c in chains["dev"]])
        probs[(var, "dev")] = {c["video_id"]: oof[off[i]:off[i + 1]] for i, c in enumerate(chains["dev"])}
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[y >= 0], y[y >= 0])
        for s in ("test", "val7"):
            probs[(var, s)] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in chains[s]}
    np.savez_compressed(OUT / "probs.npz", **{f"{var}|{s}|{v}": p for (var, s), d in probs.items() for v, p in d.items()},
                        **{f"label|{s}|{c['video_id']}": c["label"] for s in chains for c in chains[s]},
                        **{f"chain|{s}|{c['video_id']}": c["chains"] for s in chains for c in chains[s]})
    (OUT / "folds.json").write_text(json.dumps(fold_of, indent=1), encoding="utf-8")
    rep = {"AP": {}, "AP_diff": {}, "grid": {}, "budget": {}, "counts": {}}
    rng = np.random.default_rng(1)
    for s in ("dev", "test", "val7"):
        lab = {c["video_id"]: c["label"] for c in chains[s]}; vids = list(lab)
        rep["counts"][s] = {"chains": int(sum(len(l) for l in lab.values())), "false": int(sum((l == 1).sum() for l in lab.values())),
                            "keep": int(sum((l == 0).sum() for l in lab.values())), "uncounted": int(sum((l < 0).sum() for l in lab.values()))}

        def ap(var, vs):
            y = np.concatenate([lab[v] for v in vs]); p = np.concatenate([probs[(var, s)][v] for v in vs]); k = y >= 0
            return float(average_precision_score(y[k], p[k]))
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
        print(f"[AP] {s}: без мухи {rep['AP'][f'nofly|{s}']:.4f}, с мухой {rep['AP'][f'fly|{s}']:.4f}, разность {rep['AP_diff'][s]['mean']:+.4f} [{rep['AP_diff'][s]['ci95'][0]:+.4f}; {rep['AP_diff'][s]['ci95'][1]:+.4f}]", flush=True)
        for var in VARS:
            rep["grid"][f"{var}|{s}"] = {str(t): {k: v for k, v in losses(chains[s], probs[(var, s)], t).items() if k != "per_video"} for t in GRID}
    for var in VARS:
        for b in BUDGETS:
            ok = [t for t in GRID if rep["grid"][f"{var}|dev"][str(t)]["lost_share"] is not None and rep["grid"][f"{var}|dev"][str(t)]["lost_share"] <= b]
            thr = max(ok, key=lambda t: (rep["grid"][f"{var}|dev"][str(t)]["fp_removed"], t)) if ok else None
            rep["budget"][f"{var}|{b}"] = {"threshold": thr, **{s: (rep["grid"][f"{var}|{s}"][str(thr)] if thr is not None else None) for s in ("dev", "test", "val7")}}
            r = rep["budget"][f"{var}|{b}"]
            if thr is None:
                print(f"[BUDGET] {var} B={b:.1%}: допустимого порога на dev нет", flush=True); continue
            print(f"[BUDGET] {var} B={b:.1%}: порог {thr}; " + "; ".join(f"{s}: ложных рамок убрано {r[s]['fp_removed']}, верных {r[s]['tp_removed']}, потеряно {r[s]['lost']}/{r[s]['hit']} ({r[s]['lost_share']:.2%}), точность удаления {r[s]['precision'] if r[s]['precision'] is None else round(r[s]['precision'], 3)}" for s in ("dev", "test", "val7")), flush=True)
    (OUT / "report_rank.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK rank]")
    return 0


def stage_hota() -> int:
    from evaluation.hota_eval import evaluate_hota
    with open(OUT / "chains.pkl", "rb") as fh:
        chains = pickle.load(fh)["chains"]
    rep_r = json.loads((OUT / "report_rank.json").read_text(encoding="utf-8"))
    z = np.load(OUT / "probs.npz")
    cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
    rep = {"pooled": {}, "diff": {}, "loss": {}}
    rng = np.random.default_rng(1)
    for s in ("test", "val7"):
        vids = [c["video_id"] for c in chains[s]]; split = chains[s][0]["split"]

        def run(tag, drop_of):
            tasks = [(c["video_id"], c["out_dir"], c["split"], drop_of(c), str(OUT / "filtered" / tag / s)) for c in chains[s]]
            with ProcessPoolExecutor(max_workers=6) as pool:
                return {r["video_id"]: r for r in pool.map(apply_filter, tasks)}
        res = {"none": run("none", lambda c: np.zeros(0, int))}
        for var in VARS:
            for b in BUDGETS:
                thr = rep_r["budget"][f"{var}|{b}"]["threshold"]
                if thr is None:
                    continue
                res[f"{var}@{b}"] = run(f"{var}_{b}", lambda c, var=var, thr=thr: c["chains"][z[f"{var}|{s}|{c['video_id']}"] > thr])
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        for key in res:
            tag = "none" if key == "none" else key.replace("@", "_")
            for g, gv in groups.items():
                if gv:
                    agg = evaluate_hota(OUT / "filtered" / tag / s, gv, split=split)
                    rep["pooled"][f"{key}|{s}|{g}"] = {k: agg.get(k) for k in KEYS}
                    rep["loss"][f"{key}|{s}|{g}"] = {k: int(sum(res[key][v][k] for v in gv)) for k in ("removed_tp_boxes", "lost_gt_objects", "gt_objects_hit")}
        cmp_ = [(f"{v}@{b}", "none") for v in VARS for b in BUDGETS] + [(f"fly@{b}", f"nofly@{b}") for b in BUDGETS]
        for a_, b_ in cmp_:
            if a_ not in res or b_ not in res:
                continue
            for g, gv in groups.items():
                if not gv:
                    continue
                for met in ("HOTA", "DetA", "AssA"):
                    d = np.array([res[a_][v]["metrics"][met] - res[b_][v]["metrics"][met] for v in gv])
                    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    rep["diff"][f"{a_}-{b_}|{s}|{g}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        for key in res:
            P = rep["pooled"][f"{key}|{s}|all"]; L = rep["loss"][f"{key}|{s}|all"]
            line = f"[HOTA] {s} {key:12s}: HOTA {P['HOTA']:.3f} DetA {P['DetA']:.2f} AssA {P['AssA']:.2f} FP {P['CLR_FP']} FN {P['CLR_FN']}; потеряно {L['lost_gt_objects']}/{L['gt_objects_hit']}"
            if key.startswith("fly@") and f"{key}-nofly@{key.split('@')[1]}|{s}|all|HOTA" in rep["diff"]:
                d = rep["diff"][f"{key}-nofly@{key.split('@')[1]}|{s}|all|HOTA"]; line += f" | с мухой − без мухи по видео {d['mean_per_video']:+.3f} [{d['ci95'][0]:+.3f}; {d['ci95'][1]:+.3f}] {d['better']}/{d['n']}"
            print(line, flush=True)
        (OUT / "report_hota.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK hota]")
    return 0


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--hota", action="store_true"); ap_.add_argument("--fixed", action="store_true"); a_ = ap_.parse_args()
    if a_.fixed:
        FIXED = True; OUT = OUT.parent / (OUT.name + "_fixed")
    raise SystemExit(stage_hota() if a_.hota else stage_rank())
