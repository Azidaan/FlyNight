"""Голова «удалить отдельную жизнь» в готовом выходе после MASA — без мухи и с мухой (карточка life_filter_head_v1).

life_filter_oracle_v1: удаление полностью ложных жизней внутри цепочек (оракул) — +0,33 HOTA сверх удаления цепочек на dev.
Здесь — обучаемая голова для действия «удалить рамки жизни» (жизнь — непрерывный фрагмент трекера до сшивки), номера и
связи готового выхода не меняются.
  Метка жизни — правило output_filter_oracle по опорным кадрам: 1 — все учитываемые рамки ложные, 0 — есть верная,
  −1 — нет учитываемых рамок (не обучается и не удаляется: на HOTA не влияет). Метка зависит только от рамок кадра,
  а не от номеров, поэтому для train 130 берётся с выхода перекрёстной сшивки (runs/chain_filter_train130/repro/train).
  Принадлежность рамки выхода к жизни — по координатам строки трекера того же кадра (как в life_filter_oracle).
  Признаки жизни (scripts/stitch_realness.life_features): без мухи — длина, число строк, уверенность детектора, рамки
  начала и конца, класс, вход глаза (среднее начала и конца) — 751; с мухой — плюс сигнатуры мухи начала и конца (1 292).
  Голова — логистическая регрессия C = 0,1, обучение на жизнях train 130 с меткой. dev — проверка: AP, бутстрэп по видео;
  порог при бюджете потерь 1 % объектов (наибольшее число удалённых ложных рамок). Критерий перехода (до прогона):
  AP с мухой − без мухи на dev > 0, ДИ > 0. --open: HOTA при бюджете 1 % на dev (порог из dev), test и val7.

Режим --v2 (life_filter_head_v2, исправление 09.10): жизнь — запись в signatures.npz (номер трекера + интервал), рамка
относится к записи по номеру и кадру (scripts/life_segments.py); признаки — строки трекера только внутри интервала записи
(life_features(by_interval=True)); удаляется запись, а не номер. Выход — runs/life_filter_head_v2.

Запуск: python scripts/life_filter_head.py [--v2] [--open]
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

OUT = PROJECT_ROOT / "runs/life_filter_head"
SRC = {"train": PROJECT_ROOT / "runs/chain_filter_train130/repro/train", "dev": PROJECT_ROOT / "runs/chain_filter_rank/repro/dev",
       "test": PROJECT_ROOT / "runs/chain_filter_rank/repro/test", "val7": PROJECT_ROOT / "runs/chain_filter_rank/repro/val7"}
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
KEYS = ("HOTA", "DetA", "AssA", "CLR_TP", "CLR_FP", "CLR_FN")


def box_lives(vid, sig_dir, root, res):
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    o = np.load(Path(sig_dir) / vid / "signatures.npz")["outputs"]; by = {}
    for r in o:
        by.setdefault(int(r[0]), []).append(r)
    life = []
    for i, fr in enumerate(res):
        rows = by.get(int(matches[i]["video_frame_index"]), []); cur = []
        for l in fr.get("labels") or []:
            b = l["box2d"]; key = np.array([b["x1"], b["y1"], b["x2"], b["y2"]])
            hit = [int(r[5]) for r in rows if np.allclose(r[1:5], key, atol=1e-2)]
            cur.append(hit[0] if len(hit) == 1 else None)
        life.append(cur)
    return life


def video_task(task):
    from scripts.chain_filter_head import classify_ids
    from scripts.stitch_realness import life_features
    vid, sig_dir, root, split, s, v2 = task
    res = json.loads((SRC[s] / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    segstats = None
    if v2:
        from scripts.life_segments import box_segments
        life, segstats = box_segments(vid, sig_dir, root, res)
    else:
        life = box_lives(vid, sig_dir, root, res)
    tags = classify_ids(res, gt)
    z = np.load(Path(sig_dir) / vid / "signatures.npz"); tids = z["track_id"].astype(int)
    pos = {i: i for i in range(len(tids))} if v2 else {t: i for i, t in enumerate(tids)}   # v2: ключ рамки — номер записи жизни
    n = len(tids); n_tp = np.zeros(n, np.int32); n_fp = np.zeros(n, np.int32); tp_map = {}
    for tg, lv in zip(tags, life):
        for (t, g), tid in zip(tg, lv):
            if tid is None or tid not in pos:
                continue
            i = pos[tid]
            if t == "tp":
                n_tp[i] += 1; tp_map[(i, g)] = tp_map.get((i, g), 0) + 1
            elif t == "fp":
                n_fp[i] += 1
    label = np.where(n_tp > 0, 0, np.where(n_fp > 0, 1, -1)).astype(np.int8)
    f = life_features(str(Path(sig_dir) / vid / "signatures.npz"), by_interval=v2)
    nofly = np.concatenate([f["common"], f["cls"], f["end"], f["start"], (f["hex_end"] + f["hex_start"]) / 2], 1).astype(np.float32)
    fly = np.concatenate([nofly, f["fly_end"], f["fly_start"]], 1).astype(np.float32)
    return {"video_id": vid, "split": split, "set": s, "sig_dir": sig_dir, "root": root, "tids": tids, "label": label, "n_tp": n_tp, "n_fp": n_fp,
            "tp_map": [(i, g, k) for (i, g), k in tp_map.items()], "nofly": nofly, "fly": fly, "unmatched": int(sum(x is None for lv in life for x in lv)), "segstats": segstats}


def losses(vs, probs, thr):
    out = {"removed_counted": 0, "removed_false": 0, "fp_removed": 0, "tp_removed": 0, "lost": 0, "hit": 0}
    for c in vs:
        rm = probs[c["video_id"]] > thr; tot, gone = {}, {}
        for i, g, k in c["tp_map"]:
            tot[g] = tot.get(g, 0) + k
            if rm[i]:
                gone[g] = gone.get(g, 0) + k
        out["removed_counted"] += int((rm & (c["label"] >= 0)).sum()); out["removed_false"] += int((rm & (c["label"] == 1)).sum())
        out["fp_removed"] += int(c["n_fp"][rm].sum()); out["tp_removed"] += int(c["n_tp"][rm].sum())
        out["lost"] += sum(1 for g in tot if gone.get(g, 0) == tot[g]); out["hit"] += len(tot)
    out["precision"] = out["removed_false"] / out["removed_counted"] if out["removed_counted"] else None
    out["lost_share"] = out["lost"] / out["hit"] if out["hit"] else None
    return out


def apply_task(task):
    from evaluation.hota_eval import evaluate_hota
    from scripts.chain_filter_head import classify_ids
    from scripts.run_bdd_video_rate_experiment import save_json
    vid, sig_dir, root, split, s, drop_tids, out_dir, v2 = task
    res = json.loads((SRC[s] / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    if v2:
        from scripts.life_segments import box_segments
        life = box_segments(vid, sig_dir, root, res)[0]
    else:
        life = box_lives(vid, sig_dir, root, res)
    drop = set(int(t) for t in drop_tids)   # v2 — номера записей жизней, v1 — номера трекера
    before = {g for tg in classify_ids(res, gt) for t, g in tg if t == "tp"}
    rows = [{**fr, "labels": [l for l, tid in zip(fr.get("labels") or [], lv) if tid is None or tid not in drop]} for fr, lv in zip(res, life)]
    after = {g for tg in classify_ids(rows, gt) for t, g in tg if t == "tp"}
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
    m = evaluate_hota(d, [vid], split=split)
    return {"video_id": vid, "metrics": {k: m.get(k) for k in KEYS}, "lost": len(before - after), "hit": len(before)}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.fly_lifecycle_alive import sets
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--open", action="store_true"); ap_.add_argument("--v2", action="store_true"); a = ap_.parse_args()
    global OUT
    if a.v2:
        OUT = OUT.parent / (OUT.name + "_v2")
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    cache = OUT / "lives.pkl"
    if cache.exists():
        L = pickle.load(open(cache, "rb"))
    else:
        L = {}
        with ProcessPoolExecutor(max_workers=6) as pool:
            for s in ("train", "dev", "test", "val7"):
                L[s] = list(pool.map(video_task, [(v, str(sr), str(rt), sp, s, a.v2) for v, sr, rt, sp in S[s]]))
                lab = np.concatenate([c["label"] for c in L[s]])
                print(f"[LIVES] {s}: жизней {len(lab)}, полностью ложных {(lab == 1).sum()}, с верной рамкой {(lab == 0).sum()}, без учитываемых {(lab < 0).sum()}; рамок без жизни {sum(c['unmatched'] for c in L[s])}"
                      + (f"; сопоставление {dict((k, sum(c['segstats'][k] for c in L[s])) for k in L[s][0]['segstats'])}" if a.v2 else ""), flush=True)
        pickle.dump(L, open(cache, "wb"))
    sets_ = ("dev", "test", "val7") if a.open else ("dev",)
    probs = {}
    ytr = np.concatenate([c["label"] for c in L["train"]]); ok = ytr >= 0
    for var in ("nofly", "fly"):
        X = np.concatenate([c[var] for c in L["train"]])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[ok], ytr[ok])
        for s in sets_:
            probs[(var, s)] = {c["video_id"]: m.predict_proba(c[var])[:, 1] for c in L[s]}
    np.savez_compressed(OUT / ("probs_open.npz" if a.open else "probs_dev.npz"), **{f"{var}|{s}|{v}": p for (var, s), d in probs.items() for v, p in d.items()})
    rep = {"AP": {}, "AP_diff": {}, "budget": {}, "gate": None, "train_labeled": int(ok.sum())}
    rng = np.random.default_rng(1)
    for s in sets_:
        lab = {c["video_id"]: c["label"] for c in L[s]}; vids = list(lab)

        def ap(var, vs):
            y = np.concatenate([lab[v] for v in vs]); p = np.concatenate([probs[(var, s)][v] for v in vs]); k = y >= 0
            return float(average_precision_score(y[k], p[k]))
        for var in ("nofly", "fly"):
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
    for var in ("nofly", "fly"):
        grid = {t: losses(L["dev"], probs[(var, "dev")], t) for t in GRID}
        okt = [t for t in GRID if grid[t]["lost_share"] is not None and grid[t]["lost_share"] <= 0.01]
        thr = max(okt, key=lambda t: (grid[t]["fp_removed"], t)) if okt else None
        rep["budget"][var] = {"threshold": thr, "dev_grid": {str(t): grid[t] for t in GRID}}
        if thr is not None:
            for s in sets_:
                rep["budget"][var][s] = losses(L[s], probs[(var, s)], thr)
            r = rep["budget"][var]["dev"]
            print(f"[BUDGET 1 %] {var}: порог {thr}; dev — ложных рамок убрано {r['fp_removed']}, верных {r['tp_removed']}, потеряно {r['lost']}/{r['hit']}, точность удаления {r['precision']:.3f}", flush=True)
        else:
            print(f"[BUDGET 1 %] {var}: допустимого порога на dev нет", flush=True)
    d = rep["AP_diff"]["dev"]; rep["gate"] = bool(d["mean"] > 0 and d["ci95"][0] > 0)
    print(f"[GATE] AP с мухой − без мухи на dev {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}] → критерий перехода {'выполнен' if rep['gate'] else 'НЕ выполнен'}", flush=True)
    if a.open:
        rep["hota"], rep["hota_diff"] = {}, {}
        for s in sets_:
            split = L[s][0]["split"]; res = {}
            for var in ("none", "nofly", "fly"):
                thr = rep["budget"].get(var, {}).get("threshold")
                if var != "none" and thr is None:
                    continue
                tasks = [(c["video_id"], c["sig_dir"], c["root"], c["split"], s, (np.nonzero(probs[(var, s)][c["video_id"]] > thr)[0] if a.v2 else c["tids"][probs[(var, s)][c["video_id"]] > thr]) if var != "none" else [], str(OUT / "filtered" / var / s), a.v2) for c in L[s]]
                with ProcessPoolExecutor(max_workers=6) as pool:
                    res[var] = {r["video_id"]: r for r in pool.map(apply_task, tasks)}
                agg_ = evaluate_hota(OUT / "filtered" / var / s, list(res[var]), split=split)
                rep["hota"][f"{var}|{s}"] = {**{k: agg_.get(k) for k in KEYS}, "lost": int(sum(x["lost"] for x in res[var].values())), "hit": int(sum(x["hit"] for x in res[var].values()))}
            vids = list(res["none"])
            attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
            night = list(filter_videos_by_subset(attrs, vids, "night"))
            for x_, y_ in (("fly", "nofly"), ("nofly", "none"), ("fly", "none")):
                if x_ not in res or y_ not in res:
                    continue
                for g, gv in (("all", vids), ("night", night)):
                    if gv:
                        dd = np.array([res[x_][v]["metrics"]["HOTA"] - res[y_][v]["metrics"]["HOTA"] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                        rep["hota_diff"][f"{x_}-{y_}|{s}|{g}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "n": len(dd)}
            print(f"[HOTA] {s}: " + "; ".join(f"{v} {rep['hota'][f'{v}|{s}']['HOTA']:.3f} (потеряно {rep['hota'][f'{v}|{s}']['lost']}/{rep['hota'][f'{v}|{s}']['hit']})" for v in ("none", "nofly", "fly") if f"{v}|{s}" in rep["hota"]), flush=True)
            for k, r in rep["hota_diff"].items():
                if f"|{s}|" in k:
                    print(f"[DIFF] {k}: по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
    (OUT / ("report_open.json" if a.open else "report_dev.json")).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
