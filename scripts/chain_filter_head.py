"""Голова «удалить итоговую цепочку» на выходе после сшивки с MASA (карточка chain_filter_fly_v1).

output_filter_oracle_v1: удаление полностью ложных цепочек (оракул) поднимает HOTA выхода после MASA на +1,35 (test).
Здесь — обучаемая голова для того же действия; без мухи и с мухой, одинаковые признаки и протокол, отличие — только
признаки мухи.
  Выход — жадная сшивка «база + MASA» на пороге 0,6 (stitch_eval_v2): пересобирается по тем же вероятностям, принятые
  связи дают состав цепочек (жизни → итоговый номер); сверка — HOTA пересборки = HOTA runs/stitch_v2/.../base_masa_0.6.
  Метка цепочки — правило output_filter_oracle (опорные кадры, венгерское по классу, IoU ≥ 0,5, crowd — игнор): 1 — все
  учитываемые рамки ложные, 0 — есть верная, −1 — учитываемых рамок нет (не обучается; на HOTA не влияет).
  Признаки цепочки (по всем кадрам её жизней, runs/fly_signatures*): число жизней, длительность, число рамок, доля кадров
  с рамкой, уверенность детектора (среднее, макс., мин., σ, квартили), площадь/пропорции/положение рамки (среднее, σ),
  скорость центра в размерах рамки, класс (большинство), вход глаза (hexin, среднее по началам и концам жизней);
  с мухой — плюс сигнатура мухи (34 × 19, среднее и σ по началам и концам жизней).
  Голова — логистическая регрессия (C = 0,1, стандартизация). Обучение — dev 35; порог — по общему HOTA на dev по
  предсказаниям вне обучения (5 частей по видео) из сетки; итоговая модель — на всём dev; проверка — test 35 и val7.
  Отчёт: HOTA, DetA, AssA, CLR_TP/FP/FN; потери настоящих объектов (убранные верные рамки; объекты разметки, у которых
  исчезли все попадания); ночь/день; парные разности по видео (бутстрэп 5000): fly − nofly, nofly − без фильтра,
  fly − без фильтра; оракул — для доли реализованного запаса.

Запуск: python scripts/chain_filter_head.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402
from scripts.output_filter_oracle import VALID, _box, _iou  # noqa: E402

SRC = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/base_masa_0.6"
OUT = PROJECT_ROOT / "runs/chain_filter"
HEAD, THR = "base_masa", 0.6
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KEYS = ("HOTA", "DetA", "AssA", "CLR_TP", "CLR_FP", "CLR_FN")


def classify_ids(res: list, gt: list) -> list:
    """Как output_filter_oracle.classify, но для верной рамки возвращает id эталона: [(tag, gt_id), …] по кадрам."""
    from scipy.optimize import linear_sum_assignment
    out = []
    for fr_r, fr_g in zip(res, gt):
        R = fr_r.get("labels") or []; G = fr_g.get("labels") or []
        tags = [("skip", None) for _ in R]
        crowd = [_box(g) for g in G if (g.get("attributes") or {}).get("crowd", False)]
        for c in VALID:
            ri = [i for i, l in enumerate(R) if l.get("category") == c]
            gi = [j for j, g in enumerate(G) if g.get("category") == c and not (g.get("attributes") or {}).get("crowd", False)]
            if not ri:
                continue
            S = _iou([_box(R[i]) for i in ri], [_box(G[j]) for j in gi]); match = {}
            if S.size:
                M = S.copy(); M[M < 0.5 - np.finfo(float).eps] = 0
                rr, cc = linear_sum_assignment(-M.T)
                for g_, r_ in zip(rr, cc):
                    if M.T[g_, r_] > np.finfo(float).eps:
                        match[r_] = str(G[gi[g_]]["id"])
            ioa = _iou([_box(R[i]) for i in ri], crowd, ioa=True)
            for k, i in enumerate(ri):
                if k in match:
                    tags[i] = ("tp", match[k])
                elif ioa.size and (ioa[k] > 0.5 + np.finfo(float).eps).any():
                    tags[i] = ("ign", None)
                else:
                    tags[i] = ("fp", None)
        out.append(tags)
    return out


def chain_task(task):
    """Пересборка сшивки одного видео, состав цепочек, метки и признаки цепочек."""
    from scripts.eval_fly_reid import stitch_video
    from scripts.stitch_realness import life_features
    vid, sig_dir, root, split, split_dir, pairs, probs, out_dir = task
    r = stitch_video((vid, sig_dir, root, split, pairs, probs, THR, out_dir))
    z = np.load(Path(sig_dir) / vid / "signatures.npz"); tids = z["track_id"].astype(int)
    parent = {}

    def root_of(t):
        seen = set()
        while t in parent and t not in seen:
            seen.add(t); t = parent[t]
        return t
    for k in r["accepted"]:
        a, b = int(pairs[k, 0]), int(pairs[k, 1]); parent[root_of(int(tids[b]))] = root_of(int(tids[a]))
    chain_of_life = np.array([root_of(int(t)) for t in tids])
    res = json.loads((Path(out_dir) / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    tags = classify_ids(res, gt)
    cnt, tp_of = {}, {}
    for fr, tg in zip(res, tags):
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            d = cnt.setdefault(int(l["id"]), {"tp": 0, "fp": 0}); d[t] = d.get(t, 0) + 1
            if t == "tp":
                tp_of[(int(l["id"]), g)] = tp_of.get((int(l["id"]), g), 0) + 1
    chains = sorted(set(chain_of_life.tolist()))
    label = np.array([(1 if cnt[c]["tp"] == 0 and cnt[c]["fp"] > 0 else 0) if c in cnt and (cnt[c]["tp"] + cnt[c]["fp"]) > 0 else -1 for c in chains], np.int8)
    # признаки цепочки
    lf = life_features(str(Path(sig_dir) / vid / "signatures.npz"))
    o = z["outputs"]; rows_of = {}
    for row in o:
        rows_of.setdefault(int(row[5]), []).append(row)
    F_nofly, F_fly = [], []
    for c in chains:
        L = np.nonzero(chain_of_life == c)[0]
        R = np.concatenate([np.asarray(rows_of.get(int(tids[i]), []), np.float64).reshape(-1, 8) for i in L])
        R = R[np.argsort(R[:, 0])]
        sc = R[:, 6] if len(R) else np.zeros(1)
        w = np.clip(R[:, 3] - R[:, 1], 1, None) if len(R) else np.ones(1); h = np.clip(R[:, 4] - R[:, 2], 1, None) if len(R) else np.ones(1)
        cx = (R[:, 1] + R[:, 3]) / 2 if len(R) else np.zeros(1); cy = (R[:, 2] + R[:, 4]) / 2 if len(R) else np.zeros(1)
        span = int(z["end"][L].max()) - int(z["start"][L].min()) + 1
        sp = np.hypot(np.diff(cx), np.diff(cy)) / np.sqrt(w[:-1] * h[:-1]) if len(R) > 1 else np.zeros(1)
        cls = np.zeros(8); cls[min(int(np.bincount(z["cls"][L].astype(int)).argmax()), 7)] = 1
        base = [len(L), np.log1p(span), np.log1p(len(R)), len(R) / span, sc.mean(), sc.max(), sc.min(), sc.std(), *np.quantile(sc, [.25, .5, .75]),
                np.log(w * h).mean(), np.log(w * h).std(), np.log(w / h).mean(), (cy / 720).mean(), (cy / 720).std(), (cx / 1280).mean(), sp.mean(), sp.max(), *cls]
        hexm = np.concatenate([lf["hex_end"][L], lf["hex_start"][L]]).mean(0)
        flyv = np.concatenate([lf["fly_end"][L], lf["fly_start"][L]])
        F_nofly.append(np.concatenate([np.asarray(base, np.float32), hexm]))
        F_fly.append(np.concatenate([F_nofly[-1], flyv.mean(0), flyv.std(0)]))
    pos = {c: i for i, c in enumerate(chains)}
    n_tp = np.array([cnt.get(c, {}).get("tp", 0) for c in chains], np.int32); n_fp = np.array([cnt.get(c, {}).get("fp", 0) for c in chains], np.int32)
    tp_map = [(pos[c], g, k) for (c, g), k in tp_of.items() if c in pos]   # (цепочка, объект разметки, верных рамок) — для подсчёта потерь без оценщика
    return {"video_id": vid, "split_dir": split_dir, "chains": np.array(chains), "label": label, "n_tp": n_tp, "n_fp": n_fp, "tp_map": tp_map, "life_chain": np.searchsorted(np.array(chains), chain_of_life).astype(np.int32), "life_end": z["end"].astype(np.int32), "life_start": z["start"].astype(np.int32), "nofly": np.asarray(F_nofly, np.float32), "fly": np.asarray(F_fly, np.float32), "out_dir": out_dir, "split": split}


def apply_filter(task):
    """Удалить рамки цепочек с вероятностью «ложная» > порога; HOTA по видео; потери настоящих объектов."""
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import save_json
    vid, src_dir, split, drop_ids, out_dir = task
    res = json.loads((Path(src_dir) / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    tags = classify_ids(res, gt); drop = set(int(x) for x in drop_ids)
    rows, lost_tp, gt_before, gt_after = [], 0, set(), set()
    for fr, tg in zip(res, tags):
        keep = []
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            if t == "tp":
                gt_before.add(g)
            if int(l["id"]) in drop:
                lost_tp += int(t == "tp")
            else:
                keep.append(l)
                if t == "tp":
                    gt_after.add(g)
        rows.append({**fr, "labels": keep})
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True); save_json(d / f"{vid}.json", rows)
    m = evaluate_hota(d, [vid], split=split)
    return {"video_id": vid, "metrics": {k: m.get(k) for k in KEYS}, "removed_tp_boxes": lost_tp, "lost_gt_objects": len(gt_before - gt_after), "gt_objects_hit": len(gt_before)}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    cond = json.loads((PROJECT_ROOT / "runs/oracle_ladder/report.json").read_text(encoding="utf-8"))["cond"]
    data = {}
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        for s in ("train", "dev", "test", "val7"):
            data[s] = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    idx = [V.NAMES.index(c) for c in V.HEADS[HEAD]]
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr)
    del Xtr, ytr, data["train"]
    rep = {"repro": {}, "labels": {}, "threshold": {}, "pooled": {}, "diff": {}, "loss": {}, "share": {}}
    chains = {}
    for s in ("dev", "test", "val7"):
        meta = {v: (str(sr), str(rt), sp) for v, sr, rt, sp in S[s]}
        tasks = [(d["video_id"], *meta[d["video_id"]], s, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1], str(OUT / "repro" / s)) for d in data[s]]
        with ProcessPoolExecutor(max_workers=6) as pool:
            chains[s] = list(pool.map(chain_task, tasks))
        vids = [c["video_id"] for c in chains[s]]; sp = chains[s][0]["split"]
        rp = evaluate_hota(OUT / "repro" / s, vids, split=sp)["HOTA"]
        src = evaluate_hota(SRC / s, vids, split=sp)["HOTA"] if (SRC / s).exists() else None
        rep["repro"][s] = {"rebuilt": rp, "source": src}
        lab = np.concatenate([c["label"] for c in chains[s]])
        rep["labels"][s] = {"chains": int(len(lab)), "false": int((lab == 1).sum()), "keep": int((lab == 0).sum()), "unknown": int((lab == -1).sum())}
        print(f"[REPRO] {s}: HOTA пересборки {rp:.3f}, источник {src}; цепочек {rep['labels'][s]}", flush=True)
    del data

    def stack(s, var):
        X = np.concatenate([c[var] for c in chains[s]]); y = np.concatenate([c["label"] for c in chains[s]])
        g = np.concatenate([np.full(len(c["label"]), i) for i, c in enumerate(chains[s])]); return X, y, g

    def run_filter(s, probs_by_video, thr, tag):
        tasks = [(c["video_id"], c["out_dir"], c["split"], c["chains"][probs_by_video[c["video_id"]] > thr], str(OUT / "filtered" / tag / s)) for c in chains[s]]
        with ProcessPoolExecutor(max_workers=6) as pool:
            per = {r["video_id"]: r for r in pool.map(apply_filter, tasks)}
        vids = list(per)
        agg = evaluate_hota(OUT / "filtered" / tag / s, vids, split=chains[s][0]["split"])
        per["__pooled__"] = {k: agg.get(k) for k in KEYS}
        return per

    rng = np.random.default_rng(1)
    res = {}
    for var in ("nofly", "fly"):
        X, y, g = stack("dev", var); oof = np.zeros(len(y))
        folds = np.arange(len(chains["dev"])) % 5
        rng_f = np.random.default_rng(7); rng_f.shuffle(folds)
        for f in range(5):
            tr = (folds[g] != f) & (y >= 0); te = folds[g] == f
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        oof_v = {c["video_id"]: oof[g == i] for i, c in enumerate(chains["dev"])}
        grid = {t: run_filter("dev", oof_v, t, f"{var}_dev_{t:g}") for t in GRID}
        thr = max(grid, key=lambda t: grid[t]["__pooled__"]["HOTA"])
        rep["threshold"][var] = {"chosen": thr, "dev_grid": {str(t): grid[t]["__pooled__"]["HOTA"] for t in GRID}}
        print(f"[THR] {var}: порог {thr}; dev HOTA по сетке " + ", ".join(f"{t}: {grid[t]['__pooled__']['HOTA']:.3f}" for t in GRID), flush=True)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X[y >= 0], y[y >= 0])
        for s in ("test", "val7"):
            Xs, ys, gs = stack(s, var); p = m.predict_proba(Xs)[:, 1]
            res[(var, s)] = run_filter(s, {c["video_id"]: p[gs == i] for i, c in enumerate(chains[s])}, thr, f"{var}_{thr:g}")
    for s in ("test", "val7"):
        res[("none", s)] = run_filter(s, {c["video_id"]: np.zeros(len(c["chains"])) for c in chains[s]}, 2.0, "none")
        res[("oracle", s)] = run_filter(s, {c["video_id"]: (c["label"] == 1).astype(float) for c in chains[s]}, 0.5, "oracle")
        vids = [c["video_id"] for c in chains[s]]
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        for var in ("none", "nofly", "fly", "oracle"):
            P = res[(var, s)]
            rep["pooled"][f"{var}|{s}|all"] = P["__pooled__"]
            rep["loss"][f"{var}|{s}"] = {k: int(sum(P[v][k] for v in vids)) for k in ("removed_tp_boxes", "lost_gt_objects", "gt_objects_hit")}
            for gname in ("night", "day"):
                if groups[gname]:
                    rep["loss"][f"{var}|{s}|{gname}"] = {k: int(sum(P[v][k] for v in groups[gname])) for k in ("removed_tp_boxes", "lost_gt_objects", "gt_objects_hit")}
        for gname, gv in groups.items():
            if not gv:
                continue
            if gname != "all":
                for var in ("none", "nofly", "fly", "oracle"):
                    tag = {"none": "none", "oracle": "oracle"}.get(var, f"{var}_{rep['threshold'].get(var, {}).get('chosen', 0):g}")
                    agg = evaluate_hota(OUT / "filtered" / tag / s, gv, split=chains[s][0]["split"])
                    rep["pooled"][f"{var}|{s}|{gname}"] = {k: agg.get(k) for k in KEYS}
            for a_, b_ in (("fly", "nofly"), ("nofly", "none"), ("fly", "none"), ("oracle", "none")):
                for met in ("HOTA", "DetA", "AssA"):
                    d = np.array([res[(a_, s)][v]["metrics"][met] - res[(b_, s)][v]["metrics"][met] for v in gv])
                    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    rep["diff"][f"{a_}-{b_}|{s}|{gname}|{met}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
        H = lambda var: rep["pooled"][f"{var}|{s}|all"]["HOTA"]
        den = H("oracle") - H("none")
        for var in ("nofly", "fly"):
            rep["share"][f"{var}|{s}"] = (H(var) - H("none")) / den if den > 0 else None
        print(f"[{s}] HOTA общая: без фильтра {H('none'):.3f}, без мухи {H('nofly'):.3f}, с мухой {H('fly'):.3f}, оракул {H('oracle'):.3f}; доля запаса без мухи {rep['share'][f'nofly|{s}']}, с мухой {rep['share'][f'fly|{s}']}", flush=True)
        for a_, b_ in (("fly", "nofly"), ("nofly", "none"), ("fly", "none")):
            r = rep["diff"][f"{a_}-{b_}|{s}|all|HOTA"]
            print(f"[{s}] {a_} − {b_}: HOTA по видео {r['mean_per_video']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
        print(f"[{s}] потери: " + "; ".join(f"{var}: верных рамок убрано {rep['loss'][f'{var}|{s}']['removed_tp_boxes']}, объектов потеряно {rep['loss'][f'{var}|{s}']['lost_gt_objects']} из {rep['loss'][f'{var}|{s}']['gt_objects_hit']}" for var in ("nofly", "fly", "oracle")), flush=True)
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
