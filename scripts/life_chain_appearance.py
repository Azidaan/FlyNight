"""Связь с цепочкой и внешность MASA поверх соседства — H1 / B1 / B2 (карточка life_chain_appearance_v1, вариант (б)).

Последняя дешёвая проверка ветки удаления жизней (решение автора 10.10). Протокол как в life_context_neighbors_v1.
  H1 — нынешние признаки без мухи + соседство (runs/life_context_neighbors/context.pkl);
  B1 — H1 + признаки цепочки и внешности; B2 — B1 + муха (сигнатуры начала и конца).
Признаки цепочки — из выхода сшивки «база + MASA», порог 0,6 (scripts/life_filter_head.SRC: train — перекрёстная сшивка,
dev — сшивка головой пары, обученной на train); цепочка записи — номер её рамок в выходе (scripts/life_segments.box_segments):
  признак наличия, число жизней цепочки, число рамок цепочки (log), длительность цепочки (log), доля рамок записи, одиночная
  цепочка, положение записи (первая / внутренняя / последняя), средняя уверенность рамок цепочки.
Признаки внешности (runs/masa_embeddings, эмбеддинги начала и конца записи): косинус начала записи с концом предыдущей записи
цепочки и конца записи с началом следующей (+ признаки наличия); косинус с предшественником и преемником по правилу
соседства (≤ 30 кадров, ≤ 2 размеров рамки). Всё наблюдаемо, без разметки.
Порог — то же правило: 5 частей train 130 вне обучения, допустимость при потере объектов ≤ 1 % и верных рамок ≤ 0,5 % в каждой
части, наибольшая средняя разность HOTA с ДИ > 0, иначе «ничего не удалять». dev 35 — проверка.
Критерий: AP B2 − B1 > 0 (ДИ > 0) И HOTA B2 − B1 > 0 (ДИ > 0). Описательно: B1 − H1, ночь.

Запуск: python scripts/life_chain_appearance.py
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
from scripts.life_filter_head import KEYS, SRC, apply_task, losses  # noqa: E402

LIVES = PROJECT_ROOT / "runs/life_filter_head_v2/lives.pkl"
CTX = PROJECT_ROOT / "runs/life_context_neighbors/context.pkl"
OUT = PROJECT_ROOT / "runs/life_chain_appearance"
GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, None]
MAX_LOST, MAX_TP = 0.01, 0.005
NCA = 16


def chain_app(task) -> np.ndarray:
    from scripts.life_segments import box_segments
    vid, sig_dir, root, s = task
    res = json.loads((SRC[s] / f"{vid}.json").read_text(encoding="utf-8"))
    seg, _ = box_segments(vid, sig_dir, root, res)
    z = np.load(Path(sig_dir) / vid / "signatures.npz"); n = len(z["track_id"]); st = z["start"].astype(int); en = z["end"].astype(int)
    me = np.load(PROJECT_ROOT / "runs/masa_embeddings" / f"{vid}.npz"); mf, ml = me["first"].astype(np.float32), me["last"].astype(np.float32)
    chain_of = {}; boxes_of = np.zeros(n, int); score = {}
    for fr, sv in zip(res, seg):
        for l, k in zip(fr.get("labels") or [], sv):
            if k is None:
                continue
            chain_of.setdefault(k, {}).setdefault(str(l["id"]), 0); chain_of[k][str(l["id"])] += 1
            boxes_of[k] += 1; score.setdefault(str(l["id"]), []).append(float(l.get("score", 0.0)))
    rec_chain = {k: max(d, key=d.get) for k, d in chain_of.items()}
    members = {}
    for k, c in rec_chain.items():
        members.setdefault(c, []).append(k)
    fb, lb = z["first_box"].astype(np.float64), z["last_box"].astype(np.float64)
    size = np.sqrt(np.maximum((fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1]), 1.0)); size_l = np.sqrt(np.maximum((lb[:, 2] - lb[:, 0]) * (lb[:, 3] - lb[:, 1]), 1.0))
    cf = (fb[:, :2] + fb[:, 2:]) / 2; cl = (lb[:, :2] + lb[:, 2:]) / 2
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    F = np.zeros((n, NCA), np.float32)
    for i in range(n):
        f = []
        c = rec_chain.get(i)
        if c is None:
            f += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            prev = nxt = None
        else:
            mem = members[c]; tot = sum(boxes_of[m] for m in mem); span = max(en[m] for m in mem) - min(st[m] for m in mem) + 1
            first = min(mem, key=lambda m: st[m]); last = max(mem, key=lambda m: en[m])
            f += [1, len(mem), np.log1p(tot), np.log1p(span), boxes_of[i] / max(tot, 1), float(len(mem) == 1), float(i == first), float(i != first and i != last), float(i == last), float(np.mean(score.get(c, [0.0])))]
            before = [m for m in mem if en[m] < st[i]]; after = [m for m in mem if st[m] > en[i]]
            prev = max(before, key=lambda m: en[m]) if before else None; nxt = min(after, key=lambda m: st[m]) if after else None
        f += [float(prev is not None), cos(mf[i], ml[prev]) if prev is not None else 0.0, float(nxt is not None), cos(ml[i], mf[nxt]) if nxt is not None else 0.0]
        gap_p = st[i] - en; dp = np.linalg.norm(cl - cf[i], axis=1) / size[i]
        cand = np.nonzero((gap_p > 0) & (gap_p <= 30) & (dp <= 2) & (np.arange(n) != i))[0]
        f.append(cos(mf[i], ml[cand[np.lexsort((dp[cand], gap_p[cand]))[0]]]) if len(cand) else 0.0)
        gap_s = st - en[i]; ds = np.linalg.norm(cf - cl[i], axis=1) / size_l[i]
        cand = np.nonzero((gap_s > 0) & (gap_s <= 30) & (ds <= 2) & (np.arange(n) != i))[0]
        f.append(cos(ml[i], mf[cand[np.lexsort((ds[cand], gap_s[cand]))[0]]]) if len(cand) else 0.0)
        F[i] = f
    return F


def main() -> int:
    from sklearn.metrics import average_precision_score
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    OUT.mkdir(parents=True, exist_ok=True)
    L = pickle.load(open(LIVES, "rb")); tr, dv = L["train"], L["dev"]; ctx = pickle.load(open(CTX, "rb"))
    cache = OUT / "chain_app.pkl"
    if cache.exists():
        ca = pickle.load(open(cache, "rb"))
    else:
        vids = tr + dv
        with ProcessPoolExecutor(max_workers=6) as pool:
            ca = dict(zip([c["video_id"] for c in vids], pool.map(chain_app, [(c["video_id"], c["sig_dir"], c["root"], c["set"]) for c in vids])))
        pickle.dump(ca, open(cache, "wb"))
    print(f"[CA] признаки цепочки и внешности для {len(ca)} видео", flush=True)
    feat = {"H1": lambda c: np.concatenate([c["nofly"], ctx[c["video_id"]]], 1),
            "B1": lambda c: np.concatenate([c["nofly"], ctx[c["video_id"]], ca[c["video_id"]]], 1),
            "B2": lambda c: np.concatenate([c["fly"], ctx[c["video_id"]], ca[c["video_id"]]], 1)}
    cfg = CONFIGS[0]
    fold = np.arange(len(tr)) % 5; np.random.default_rng(5).shuffle(fold)   # те же части, что в life_context_neighbors_v1
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
        print(f"[FIT] {hname} готова", flush=True)
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
        for a_, b_ in (("B2", "B1"), ("B1", "H1")):
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
        for k in res:
            agg = evaluate_hota(OUT / "filtered" / ("dev_none" if k == "none" else f"dev_{k}") / "dev", gv, split="train")
            rep["pooled"][f"{k}|{g}"] = {m: agg.get(m) for m in KEYS}
        for a_, b_ in (("B2", "B1"), ("B1", "H1"), ("H1", "none"), ("B1", "none"), ("B2", "none")):
            d = np.array([res[a_][v]["metrics"]["HOTA"] - res[b_][v]["metrics"]["HOTA"] for v in gv]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{a_}-{b_}|{g}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}
    ci = lambda d: d["ci95"][0] > 0
    rep["criteria"] = {"AP_B2_minus_B1": bool(ci(rep["AP_diff"]["B2-B1|all"])), "HOTA_B2_minus_B1": bool(ci(rep["diff"]["B2-B1|all"]))}
    rep["criteria"]["fly_contribution"] = rep["criteria"]["AP_B2_minus_B1"] and rep["criteria"]["HOTA_B2_minus_B1"]
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
