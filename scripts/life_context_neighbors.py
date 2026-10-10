"""Контекст соседних треков для различения ложных жизней (карточка life_context_neighbors_v1, вариант (а)).

Признаки соседства — только наблюдаемые, из выхода трекера (signatures.npz → outputs: кадр, рамка, номер, уверенность,
класс) по интервалу записи жизни; от сшивки и разметки не зависят.
  одновременные: ov_max, ov_mean — наибольшее и среднее по кадрам наибольшее IoU с рамками других записей в том же кадре;
    n_ov — среднее число рамок с IoU > 0,3; у наиболее перекрывающего соседа — длина записи (log), средняя уверенность,
    совпадение класса, признак наличия; dens — среднее число рамок в кадре;
  предшественник/преемник: ближайшая запись, окончившаяся ≤ 30 кадров до начала (начавшаяся ≤ 30 кадров после конца), центр
    её крайней рамки в пределах 2 размеров рамки этой жизни: признак наличия, разрыв/30, расстояние в размерах, длина (log),
    средняя уверенность, тот же номер трекера.
Головы — HistGradientBoosting, конфигурация 0 (scripts/life_filter_boost.py): H0 — нынешние признаки без мухи; H1 — H0 +
соседство; H2 — H1 + муха. Обучение — жизни train 130 (runs/life_filter_head_v2/lives.pkl; метки train — с перекрёстной
сшивки). Порог — правило с ограничениями (как life_filter_rule_cv) по 5 частям внутри train 130 (прогнозы вне обучения):
допустим, если в каждой части потеря объектов ≤ 1 % и верных рамок ≤ 0,5 %; из допустимых — наибольшая средняя разность
HOTA с без фильтра при ДИ > 0, иначе «ничего не удалять». dev 35 — проверка: AP, HOTA, ночь отдельно.
Критерии: C1 AP H1 − H0 > 0 (ДИ > 0); C2 HOTA H1 − без фильтра > 0 (ДИ > 0); C3 AP H2 − H1 > 0 и HOTA H2 − H1 > 0 (ДИ > 0).

Запуск: python scripts/life_context_neighbors.py
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

from scripts.life_filter_boost import CONFIGS, model  # noqa: E402
from scripts.life_filter_head import KEYS, apply_task, losses  # noqa: E402

LIVES = PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl"
OUT = PROJECT_ROOT / "runs/life_context_neighbors"
GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, None]
MAX_LOST, MAX_TP = 0.01, 0.005
NCTX = 22


def _iou(b, B):
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1]); x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6); bb = np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None)
    return inter / np.maximum(a + bb - inter, 1e-6)


def context(sig_path: str) -> np.ndarray:
    z = np.load(sig_path); tid = z["track_id"].astype(int); st = z["start"].astype(int); en = z["end"].astype(int); cls = z["cls"].astype(int); o = z["outputs"]
    n = len(tid)
    recs = {}
    for i, t in enumerate(tid):
        recs.setdefault(int(t), []).append(i)
    rec_of_row = np.full(len(o), -1, int)
    for k, r in enumerate(o):
        for i in recs.get(int(r[5]), []):
            if st[i] <= r[0] <= en[i]:
                rec_of_row[k] = i; break
    length = en - st + 1
    conf = np.zeros(n)
    for i in range(n):
        m = rec_of_row == i
        conf[i] = o[m, 6].mean() if m.any() else 0.0
    by_frame = {}
    for k, r in enumerate(o):
        by_frame.setdefault(int(r[0]), []).append(k)
    F = np.zeros((n, NCTX), np.float32)
    rows_of = {}
    for k, i in enumerate(rec_of_row):
        if i >= 0:
            rows_of.setdefault(i, []).append(k)
    fb, lb = z["first_box"].astype(np.float64), z["last_box"].astype(np.float64)
    size = np.sqrt(np.maximum((fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1]), 1.0)); size_l = np.sqrt(np.maximum((lb[:, 2] - lb[:, 0]) * (lb[:, 3] - lb[:, 1]), 1.0))
    cf = (fb[:, :2] + fb[:, 2:]) / 2; cl = (lb[:, :2] + lb[:, 2:]) / 2
    for i in range(n):
        ks = rows_of.get(i, [])
        ovs, nov, dens, best, best_j = [], [], [], 0.0, -1
        for k in ks:
            fr = by_frame[int(o[k, 0])]; others = [q for q in fr if rec_of_row[q] != i]
            dens.append(len(fr))
            if others:
                iou = _iou(o[k, 1:5].astype(np.float64), o[others, 1:5].astype(np.float64))
                ovs.append(float(iou.max())); nov.append(int((iou > 0.3).sum()))
                if iou.max() > best:
                    best = float(iou.max()); best_j = rec_of_row[others[int(iou.argmax())]]
            else:
                ovs.append(0.0); nov.append(0)
        nb = [best_j >= 0, np.log1p(length[best_j]) if best_j >= 0 else 0.0, conf[best_j] if best_j >= 0 else 0.0, float(cls[best_j] == cls[i]) if best_j >= 0 else 0.0]
        base = [max(ovs) if ovs else 0.0, float(np.mean(ovs)) if ovs else 0.0, float(np.mean(nov)) if nov else 0.0, *nb, float(np.mean(dens)) if dens else 0.0]
        # предшественник: запись j, окончившаяся до начала i не дальше 30 кадров, центр её последней рамки в пределах 2 размеров первой рамки i
        gap_p = st[i] - en; dp = np.linalg.norm(cl - cf[i], axis=1) / size[i]
        cand = np.nonzero((gap_p > 0) & (gap_p <= 30) & (dp <= 2) & (np.arange(n) != i))[0]
        if len(cand):
            j = cand[np.lexsort((dp[cand], gap_p[cand]))[0]]
            pred = [1.0, gap_p[j] / 30, dp[j], np.log1p(length[j]), conf[j], float(tid[j] == tid[i])]
        else:
            pred = [0.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        gap_s = st - en[i]; ds = np.linalg.norm(cf - cl[i], axis=1) / size_l[i]
        cand = np.nonzero((gap_s > 0) & (gap_s <= 30) & (ds <= 2) & (np.arange(n) != i))[0]
        if len(cand):
            j = cand[np.lexsort((ds[cand], gap_s[cand]))[0]]
            succ = [1.0, gap_s[j] / 30, ds[j], np.log1p(length[j]), conf[j], float(tid[j] == tid[i])]
        else:
            succ = [0.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        F[i] = base + pred + succ + [np.log1p(length[i]), conf[i]][: NCTX - len(base) - 12]
    return F


def main() -> int:
    from sklearn.metrics import average_precision_score
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb")); tr, dv = L["train"], L["dev"]
    cache = OUT / "context.pkl"
    if cache.exists():
        ctx = pickle.load(open(cache, "rb"))
    else:
        with ProcessPoolExecutor(max_workers=6) as pool:
            vids = tr + dv
            ctx = dict(zip([c["video_id"] for c in vids], pool.map(context, [str(Path(c["sig_dir"]) / c["video_id"] / "signatures.npz") for c in vids])))
        pickle.dump(ctx, open(cache, "wb"))
    print(f"[CTX] признаки соседства для {len(ctx)} видео", flush=True)
    feat = {"H0": lambda c: c["nofly"], "H1": lambda c: np.concatenate([c["nofly"], ctx[c["video_id"]]], 1), "H2": lambda c: np.concatenate([c["fly"], ctx[c["video_id"]]], 1)}
    cfg = CONFIGS[0]
    fold = np.arange(len(tr)) % 5; np.random.default_rng(5).shuffle(fold)
    rep = {"thresholds": {}, "train_grid": {}, "AP": {}, "AP_diff": {}, "pooled": {}, "diff": {}, "removal": {}, "criteria": {}}
    rng = np.random.default_rng(1)
    oof = {h: {} for h in feat}; dev_p = {}
    for hname, fx in feat.items():
        for f in range(5):
            part = [c for c, k in zip(tr, fold) if k != f]
            y = np.concatenate([c["label"] for c in part]); ok = y >= 0
            m = model(cfg).fit(np.concatenate([fx(c) for c in part])[ok], y[ok])
            for c, k in zip(tr, fold):
                if k == f:
                    oof[hname][c["video_id"]] = m.predict_proba(fx(c))[:, 1]
        y = np.concatenate([c["label"] for c in tr]); ok = y >= 0
        m = model(cfg).fit(np.concatenate([fx(c) for c in tr])[ok], y[ok])
        dev_p[hname] = {c["video_id"]: m.predict_proba(fx(c))[:, 1] for c in dv}
        print(f"[FIT] {hname}: вне обучения на train и итоговая модель готовы", flush=True)
    np.savez_compressed(OUT / "probs.npz", **{f"oof|{h}|{v}": p for h, d in oof.items() for v, p in d.items()}, **{f"dev|{h}|{v}": p for h, d in dev_p.items() for v, p in d.items()})

    def hota(vs, s, probs, thr, tag):
        tk = [(c["video_id"], c["sig_dir"], c["root"], c["split"], s, (np.nonzero(probs[c["video_id"]] > thr)[0] if thr is not None else []), str(OUT / "filtered" / tag / s), True) for c in vs]
        with ProcessPoolExecutor(max_workers=6) as pool:
            return {r["video_id"]: r for r in pool.map(apply_task, tk)}
    base_tr = hota(tr, "train", {}, None, "none")
    for hname in feat:
        best, best_mean, grid = "None", None, {}
        for thr in GRID[:-1]:
            per_fold = []
            for f in range(5):
                part = [c for c, k in zip(tr, fold) if k == f]
                r = losses(part, oof[hname], thr); tp_all = int(sum(int(c["n_tp"].sum()) for c in part))
                per_fold.append({"lost_share": r["lost_share"] or 0.0, "tp_share": r["tp_removed"] / max(tp_all, 1)})
            adm = all(x["lost_share"] <= MAX_LOST and x["tp_share"] <= MAX_TP for x in per_fold)
            grid[str(thr)] = {"admissible": adm, "folds": per_fold}
            if not adm:
                continue
            res = hota(tr, "train", oof[hname], thr, f"tr_{hname}_{thr}")
            d = np.array([res[c["video_id"]]["metrics"]["HOTA"] - base_tr[c["video_id"]]["metrics"]["HOTA"] for c in tr]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            grid[str(thr)].update({"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]})
            if np.quantile(bs, .025) > 0 and (best_mean is None or d.mean() > best_mean):
                best, best_mean = str(thr), float(d.mean())
        rep["train_grid"][hname] = grid; rep["thresholds"][hname] = best
        print(f"[THR] {hname}: порог по train {best}; " + "; ".join(f"{k}: {'доп' if g['admissible'] else 'недоп'}" + (f" {g['mean_per_video']:+.3f}" if 'mean_per_video' in g else "") for k, g in grid.items()), flush=True)
    lab = {c["video_id"]: c["label"] for c in dv}; vids = list(lab)
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_train.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))

    def ap(h, vs):
        y = np.concatenate([lab[v] for v in vs]); p = np.concatenate([dev_p[h][v] for v in vs]); k = y >= 0
        return float(average_precision_score(y[k], p[k]))
    for g, gv in (("all", vids), ("night", night)):
        for hname in feat:
            rep["AP"][f"{hname}|{g}"] = ap(hname, gv)
        for a_, b_ in (("H1", "H0"), ("H2", "H1")):
            bs = []
            for _ in range(2000):
                vs = [gv[i] for i in rng.integers(0, len(gv), len(gv))]
                try:
                    bs.append(ap(a_, vs) - ap(b_, vs))
                except ValueError:
                    pass
            rep["AP_diff"][f"{a_}-{b_}|{g}"] = {"mean": rep["AP"][f"{a_}|{g}"] - rep["AP"][f"{b_}|{g}"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    res = {"none": hota(dv, "dev", {}, None, "dev_none")}
    for hname in feat:
        thr = rep["thresholds"][hname]
        res[hname] = hota(dv, "dev", dev_p[hname], None if thr == "None" else float(thr), f"dev_{hname}")
        if thr != "None":
            r = losses(dv, dev_p[hname], float(thr)); rep["removal"][hname] = {k: r[k] for k in ("fp_removed", "tp_removed", "lost", "hit", "precision")}
    for g, gv in (("all", vids), ("night", night)):
        for k, per in res.items():
            agg = evaluate_hota(OUT / "filtered" / ("dev_none" if k == "none" else f"dev_{k}") / "dev", gv, split="train")
            rep["pooled"][f"{k}|{g}"] = {m: agg.get(m) for m in KEYS}
        for a_, b_ in (("H1", "none"), ("H2", "H1"), ("H0", "none"), ("H2", "none"), ("H1", "H0")):
            d = np.array([res[a_][v]["metrics"]["HOTA"] - res[b_][v]["metrics"]["HOTA"] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{a_}-{b_}|{g}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    ci = lambda d: d["ci95"][0] > 0
    rep["criteria"] = {"C1_context_AP": bool(ci(rep["AP_diff"]["H1-H0|all"])), "C2_tracker_HOTA": bool(rep["thresholds"]["H1"] != "None" and ci(rep["diff"]["H1-none|all"])),
                       "C3_fly": bool(ci(rep["AP_diff"]["H2-H1|all"]) and ci(rep["diff"]["H2-H1|all"]))}
    for g in ("all", "night"):
        print(f"[AP] {g}: " + ", ".join(f"{h} {rep['AP'][f'{h}|{g}']:.4f}" for h in feat) + "; " + "; ".join(f"{k.split('|')[0]} {d['mean']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]" for k, d in rep["AP_diff"].items() if k.endswith(f"|{g}")), flush=True)
        print(f"[HOTA] {g}: " + ", ".join(f"{k} {rep['pooled'][f'{k}|{g}']['HOTA']:.3f}" for k in res) + "; " + "; ".join(f"{k.split('|')[0]} {d['mean_per_video']:+.3f} [{d['ci95'][0]:+.3f}; {d['ci95'][1]:+.3f}] {d['better']}/{d['n']}" for k, d in rep["diff"].items() if k.endswith(f"|{g}")), flush=True)
    print(f"[REMOVAL] {rep['removal']}", flush=True)
    print(f"[CRITERIA] {rep['criteria']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
