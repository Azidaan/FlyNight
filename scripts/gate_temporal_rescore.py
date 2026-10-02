"""Г0б: пересчёт уверенности слабых кандидатов по времени без мухи (карточка temporal_rescore_5hz_v1).

Кандидаты кэша (conf ≥ 0,01) соседних размеченных кадров (5 Гц) связываются в
тубелеты: жадно по IoU того же класса, IoU ≥ --link-iou, один к одному. Признаки
кандидата: своя уверенность, размер, положение, класс, состояние кадра (контроль
«однокадровый»); прошлое звено цепочки — уверенности и IoU связей до 3 кадров назад
(«причинный»); будущее — до 2 кадров вперёд («с запаздыванием», задержка 0,4 с).

Пересчёт: правила (среднее уверенности по окну тубелета) и обучаемая голова
(HistGradientBoosting, 2 фолда по видео: учится на одном, оценивается на другом).
Метка для обучения головы — кандидат совпадает с эталоном того же класса, IoU ≥ 0,5.

Оценка — как у детектора: в каждом кадре жадное сопоставление по новой уверенности,
один к одному, IoU ≥ 0,5, тот же класс. Порог для каждого варианта подбирается так,
чтобы общее число FP равнялось числу FP исходной уверенности на пороге 0,25. Затем —
найденные ночью и днём, по возрасту объекта; бутстреп по видео.
Чистый возврат = (найдено_вариант − найдено_база) / объектов ступени ПОРОГ
(есть кандидат того же класса с IoU ≥ 0,5, но его уверенность < 0,25).

Запуск: python scripts/gate_temporal_rescore.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.birth_cost import iou_mat, load_gt_tracks  # noqa: E402

CACHE = PROJECT_ROOT / "runs" / "det_cache_ctl_960" / "val"
OUT = PROJECT_ROOT / "runs" / "gate_g0b"
NCLS = 8
PAST, FUT = 3, 2


def link(prev_box, prev_cls, box, cls, thr):
    """Жадное связывание один к одному: для каждого текущего — индекс в прошлом кадре или −1, и IoU."""
    out = np.full(len(box), -1, np.int64)
    val = np.zeros(len(box), np.float32)
    if len(prev_box) == 0 or len(box) == 0:
        return out, val
    m = np.where(cls[:, None] == prev_cls[None, :], iou_mat(box, prev_box), 0.0)
    order = np.argsort(-m, axis=None)
    used_a, used_b = set(), set()
    for flat in order:
        a, b = divmod(int(flat), m.shape[1])
        if m[a, b] < thr:
            break
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b); out[a] = b; val[a] = m[a, b]
    return out, val


def video_table(f: Path, thr: float):
    """Признаки и сведения по кандидатам одного видео + эталон."""
    per_frame, _, age_of = load_gt_tracks(f.stem)
    d = np.load(f)
    names, fidx = d["frame_names"], d["frame_idx"]
    xyxy, conf, cls = d["xyxy"].astype(np.float32), d["conf"].astype(np.float32), d["cls"].astype(np.int64)
    nf = len(names)
    idx = [np.flatnonzero(fidx == fi) for fi in range(nf)]
    prev = np.full(len(conf), -1, np.int64); piou = np.zeros(len(conf), np.float32)
    for fi in range(1, nf):
        a, b = idx[fi], idx[fi - 1]
        l, v = link(xyxy[b], cls[b], xyxy[a], cls[a], thr)
        ok = l >= 0
        prev[a[ok]] = b[l[ok]]; piou[a] = v
    nxt = np.full(len(conf), -1, np.int64); niou = np.zeros(len(conf), np.float32)
    nxt[prev[prev >= 0]] = np.flatnonzero(prev >= 0); niou[prev[prev >= 0]] = piou[prev >= 0]

    def chain(start, step_arr, k):
        cs, ious = np.zeros((len(conf), k), np.float32), np.zeros((len(conf), k), np.float32)
        cur = np.arange(len(conf)); alive = np.ones(len(conf), bool)
        for s in range(k):
            iou_here = (piou if step_arr is prev else niou)[np.where(alive, cur, 0)]
            nxt_i = np.where(alive, step_arr[np.where(alive, cur, 0)], -1)
            alive = nxt_i >= 0
            ious[:, s] = np.where(alive, iou_here, 0)
            cur = np.where(alive, nxt_i, 0)
            cs[:, s] = np.where(alive, conf[cur], 0)
        return cs, ious

    pc, pi = chain(None, prev, PAST)
    nc, ni = chain(None, nxt, FUT)
    w = xyxy[:, 2] - xyxy[:, 0]; h = xyxy[:, 3] - xyxy[:, 1]
    fr_n = np.array([len(i) for i in idx], np.float32)[fidx]
    fr_mean = np.array([conf[i].mean() if len(i) else 0 for i in idx], np.float32)[fidx]
    single = np.column_stack([conf, np.log1p(w * h), (xyxy[:, 1] + xyxy[:, 3]) / 1440.0, (xyxy[:, 0] + xyxy[:, 2]) / 2560.0,
                              np.eye(NCLS, dtype=np.float32)[np.clip(cls, 0, NCLS - 1)], fr_n, fr_mean])
    past = np.column_stack([pc, pi, (pc > 0).sum(1)])
    fut = np.column_stack([nc, ni, (nc > 0).sum(1)])
    # правила: среднее уверенности по окну тубелета (Seq-NMS-подобное усреднение)
    def mean_over(cols):
        s = conf.copy(); n = np.ones(len(conf), np.float32)
        for c in cols:
            s += c; n += (c > 0)
        return s / n
    cm = mean_over([pc[:, s] for s in range(PAST)])
    lm = mean_over([pc[:, s] for s in range(PAST)] + [nc[:, s] for s in range(FUT)])
    rules = {"rule_causal_mean": cm, "rule_lagged_mean": lm,
             "rule_causal_boost": np.maximum(conf, cm), "rule_lagged_boost": np.maximum(conf, lm)}
    # эталон и метки
    gts, label = [], np.zeros(len(conf), np.int8)
    for fi in range(nf):
        gt = per_frame.get(fi)
        if gt is None or len(gt[0]) == 0:
            gts.append(None); continue
        gb, gc, tids = gt
        a = idx[fi]
        if len(a):
            m = np.where(gc[:, None] == cls[a][None, :], iou_mat(gb, xyxy[a]), 0.0)
            label[a] = (m.max(0) >= 0.5)
            bestc = np.where(m >= 0.5, conf[a][None, :], -1.0).max(1)
        else:
            bestc = np.full(len(gb), -1.0)
        gts.append((gb, gc, np.array([age_of[(t, fi)] for t in tids]), bestc))
    v = dict(fidx=fidx, idx=idx, xyxy=xyxy, cls=cls, conf=conf, single=single, past=past, fut=fut, rules=rules, gts=gts)
    # метка — сопоставлен при жадном сопоставлении по исходной уверенности (дубликаты одного объекта — 0)
    v["label"] = greedy_eval(v, conf)[0].astype(np.int8)
    return v


def greedy_eval(v, score):
    """Для каждого кандидата: сопоставлен ли он эталону при жадном порядке по score; для эталона — score сопоставленного."""
    tp = np.zeros(len(score), bool)
    gt_score = []
    for fi, a in enumerate(v["idx"]):
        g = v["gts"][fi]
        if g is None:
            continue
        gb, gc, ages, _ = g
        gs = np.full(len(gb), -np.inf)
        if len(a):
            m = np.where(gc[:, None] == v["cls"][a][None, :], iou_mat(gb, v["xyxy"][a]), 0.0)
            used = np.zeros(len(gb), bool)
            for j in np.argsort(-score[a]):
                col = np.where(used, 0.0, m[:, j])
                k = int(col.argmax()) if len(col) else -1
                if k >= 0 and col[k] >= 0.5:
                    used[k] = True; tp[a[j]] = True; gs[k] = score[a[j]]
        gt_score.append(gs)
    return tp, gt_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--link-iou", type=float, default=0.1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--videos", type=int, default=0)
    args = ap.parse_args()
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from sklearn.ensemble import HistGradientBoostingClassifier

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE.glob("*.npz"))
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    cond_of: dict[str, str] = {}
    for cond in ("normal", "night"):
        for vv in filter_videos_by_subset(attrs, sorted(f.stem for f in files), cond):
            cond_of.setdefault(vv, cond)
    files = [f for f in files if f.stem in cond_of]
    if args.videos:
        files = files[:args.videos]
    V = {}
    for k, f in enumerate(files, 1):
        V[f.stem] = video_table(f, args.link_iou)
        if k % 20 == 0:
            print(f"[INFO] {k}/{len(files)} видео", flush=True)
    vids = sorted(V)
    fold = {vv: i % 2 for i, vv in enumerate(vids)}

    # обучаемые головы, перекрёстная подгонка по видео
    heads = {"head_single": ["single"], "head_causal": ["single", "past"], "head_lagged": ["single", "past", "fut"]}
    scores = {vv: {"base": V[vv]["conf"], **V[vv]["rules"]} for vv in vids}
    for name, parts in heads.items():
        X = {vv: np.column_stack([V[vv][p] for p in parts]) for vv in vids}
        for k in (0, 1):
            tr = [vv for vv in vids if fold[vv] != k]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
            clf.fit(np.concatenate([X[vv] for vv in tr]), np.concatenate([V[vv]["label"] for vv in tr]))
            for vv in vids:
                if fold[vv] == k:
                    scores[vv][name] = clf.predict_proba(X[vv])[:, 1].astype(np.float32)
        print(f"[INFO] голова {name} готова", flush=True)

    variants = list(scores[vids[0]])
    per = {}  # вариант -> видео -> (tp-флаги, score, gt_scores)
    for name in variants:
        per[name] = {vv: greedy_eval(V[vv], scores[vv][name]) + (scores[vv][name],) for vv in vids}
    # порог базы и её FP
    fp_base = sum(int(((~tp) & (s >= args.conf)).sum()) for tp, _, s in per["base"].values())
    thr = {"base": args.conf}
    for name in variants:
        if name == "base":
            continue
        fs = np.sort(np.concatenate([s[~tp] for tp, _, s in per[name].values()]))[::-1]
        thr[name] = float(fs[fp_base - 1]) if fp_base <= len(fs) else float(fs[-1])  # ровно fp_base FP при >=
    # счёт по видео: найдено (ночь/день; возраст 0) и объектов ступени ПОРОГ
    def counts(name, vv):
        gsl = per[name][vv][1]
        g = [x for x in V[vv]["gts"] if x is not None]
        found = sum(int((gs >= thr[name]).sum()) for gs in gsl)
        found0 = sum(int(((gs >= thr[name]) & (x[2] == 0)).sum()) for gs, x in zip(gsl, g))
        fp = int(((~per[name][vv][0]) & (per[name][vv][2] >= thr[name])).sum())
        return found, found0, fp
    table = {name: {vv: counts(name, vv) for vv in vids} for name in variants}
    n_gt = {vv: sum(len(x[0]) for x in V[vv]["gts"] if x is not None) for vv in vids}
    n_gt0 = {vv: sum(int((x[2] == 0).sum()) for x in V[vv]["gts"] if x is not None) for vv in vids}
    n_thr = {vv: sum(int(((x[3] >= 0) & (x[3] < args.conf)).sum()) for x in V[vv]["gts"] if x is not None) for vv in vids}

    def summary(vset):
        r = {"n_gt": sum(n_gt[v] for v in vset), "n_threshold_step": sum(n_thr[v] for v in vset)}
        for name in variants:
            f = sum(table[name][v][0] for v in vset); f0 = sum(table[name][v][1] for v in vset)
            fb = sum(table["base"][v][0] for v in vset); f0b = sum(table["base"][v][1] for v in vset)
            r[name] = {"found": f, "recall": f / max(r["n_gt"], 1), "fp": sum(table[name][v][2] for v in vset),
                       "net_recovered_share_of_threshold": (f - fb) / max(r["n_threshold_step"], 1),
                       "recall_age0": f0 / max(sum(n_gt0[v] for v in vset), 1), "delta_found_age0": f0 - f0b}
        return r
    groups = {"all": vids, "night": [v for v in vids if cond_of[v] == "night"], "day": [v for v in vids if cond_of[v] == "normal"]}
    res = {"fp_budget": fp_base, "thresholds": {k: float(v) for k, v in thr.items()}, "videos": len(vids),
           "point": {g: summary(vs) for g, vs in groups.items()}, "ci95": {}}
    rng = np.random.default_rng(0)
    boots: dict[str, list] = {}
    for b in range(args.boot):
        for g in ("night", "day", "all"):
            vs = groups[g]; pick = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
            nthr = sum(n_thr[v] for v in pick)
            fb = sum(table["base"][v][0] for v in pick)
            for name in variants:
                if name == "base":
                    continue
                boots.setdefault(f"{g}.{name}.net_recovered_share_of_threshold", []).append((sum(table[name][v][0] for v in pick) - fb) / max(nthr, 1))
            for a_, b_ in (("head_causal", "head_single"), ("head_lagged", "head_single")):
                boots.setdefault(f"{g}.{a_}_minus_{b_}", []).append((sum(table[a_][v][0] for v in pick) - sum(table[b_][v][0] for v in pick)) / max(nthr, 1))
    res["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"бюджет FP {fp_base}; видео {len(vids)}")
    for g in ("night", "day", "all"):
        p = res["point"][g]
        print(f"== {g}: эталонов {p['n_gt']}, ступень ПОРОГ {p['n_threshold_step']}")
        for name in variants:
            q = p[name]; ci = res["ci95"].get(f"{g}.{name}.net_recovered_share_of_threshold", [0, 0])
            print(f"  {name:18s} найдено {q['found']:6d} полнота {q['recall']:.4f} FP {q['fp']:6d} чистый возврат {q['net_recovered_share_of_threshold']:+.2%} [{ci[0]:+.2%}; {ci[1]:+.2%}] возраст0 {q['recall_age0']:.4f}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
