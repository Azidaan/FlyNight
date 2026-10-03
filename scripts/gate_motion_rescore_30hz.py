"""Г0б-30: пересчёт слабых кандидатов по времени на 30 Гц с признаком «движется иначе, чем фон» (карточка motion_rescore_30hz_v1).

Зачем. Г0б (5 Гц) показал: ложные кандидаты так же устойчивы во времени, как настоящие
объекты. Отличие, которое может остаться, — движение: знак, отражение, фон движутся
в кадре как фон (по гомографии собственного движения камеры), а машина или пешеход —
своим ходом. Здесь это проверяется на сыром видео 30 Гц без мухи; муха (T4/T5) — следующий
гейт, только если этот механизм работает.

Признаки кандидата на размеченном кадре:
  single  — уверенность, размер, положение, класс, состояние кадра (контроль);
  tube    — цепочка того же кандидата по сырым кадрам (жадное связывание, IoU ≥ 0,3 после
            переноса прошлой рамки гомографией фона): уверенности и IoU связей до 8 кадров
            назад и 8 вперёд (задержка 0,27 с);
  motion  — по той же цепочке: смещение центра рамки минус смещение, предсказанное
            гомографией фона (DIS, RANSAC по всем точкам сетки), в долях диагонали и в px;
            согласованность этого остатка; изменение размера против масштаба гомографии;
            остаток DIS-векторов внутри рамки (если там есть точки сетки).
Головы: HistGradientBoosting; учатся на 130 видео train_cache «train» (только размеченные
кадры, которых нет в обучающем списке детектора full_step5), оцениваются на dev+test (70 видео,
такие же отложенные кадры; детектор видел другие кадры этих видео — смещённая выборка) и на
val7 (детектор этих видео не видел). Оценка — как в Г0б: полнота при равном FP с исходной
уверенностью на пороге 0,25 внутри каждого набора; бутстреп по видео.

Запуск: python scripts/gate_motion_rescore_30hz.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "gate_g0b30"
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
CATS = {"pedestrian": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7}
K = 8
NCLS = 8


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    sa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); sb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1e-9)


def warp_boxes(H, box):
    """Перенос рамок гомографией: по четырём углам, рамка — их охват."""
    if H is None or len(box) == 0:
        return box.copy()
    c = np.stack([box[:, [0, 1]], box[:, [2, 1]], box[:, [0, 3]], box[:, [2, 3]]], 1).reshape(-1, 2)
    ch = np.c_[c, np.ones(len(c))] @ H.T
    ch = (ch[:, :2] / ch[:, 2:3]).reshape(-1, 4, 2)
    return np.c_[ch[..., 0].min(1), ch[..., 1].min(1), ch[..., 0].max(1), ch[..., 1].max(1)].astype(np.float32)


def homographies(points, vectors):
    import cv2
    Hs = [None]
    for r in range(1, len(vectors)):
        v = vectors[r]
        ok = np.isfinite(v).all(1)
        if ok.sum() < 8:
            Hs.append(None); continue
        H, _ = cv2.findHomography(points[ok].astype(np.float32), (points[ok] + v[ok]).astype(np.float32), cv2.RANSAC, 2.0)
        Hs.append(H)
    return Hs


def load_gt(split, vid):
    data = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    frames = sorted(data, key=lambda x: x["frameIndex"])
    seen, per = {}, {}
    for fr in frames:
        b, c, ids = [], [], []
        for l in fr.get("labels") or []:
            bb, cat = l.get("box2d"), l.get("category")
            if not bb or cat not in CATS:
                continue
            b.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); c.append(CATS[cat]); ids.append(l["id"])
        ages = []
        for t in ids:
            ages.append(seen.get(t, 0)); seen[t] = seen.get(t, 0) + 1
        per[fr["name"]] = (np.array(b, np.float32).reshape(-1, 4), np.array(c, np.int64), np.array(ages))
    return per


def video_features(task):
    vid, root, split, exclude = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)
    base = Path(root) / vid
    d = np.load(base / "detections/raw_video.npz")
    xyxy, conf, cls, fidx = d["xyxy"].astype(np.float32), d["conf"].astype(np.float32), d["cls"].astype(np.int64), d["frame_idx"]
    nf = int(d["frame_count"]) if "frame_count" in d.files else int(fidx.max()) + 1
    with np.load(base / "motion/dis_quarter.npz") as z:
        P, V = z["points"].astype(np.float32), z["vectors"].astype(np.float32)
    Hs = homographies(P, V)
    al = json.loads((base / "alignment.json").read_text(encoding="utf-8"))["matches"]
    gt = load_gt(split, vid)
    idx = [np.flatnonzero(fidx == r) for r in range(nf)]
    prev = np.full(len(conf), -1, np.int64)
    for r in range(1, nf):
        a, b = idx[r], idx[r - 1]
        if len(a) == 0 or len(b) == 0:
            continue
        m = np.where(cls[a][:, None] == cls[b][None, :], iou_mat(xyxy[a], warp_boxes(Hs[r], xyxy[b])), 0.0)
        used_b = set()
        for flat in np.argsort(-m, axis=None):
            i, j = divmod(int(flat), m.shape[1])
            if m[i, j] < 0.3:
                break
            if prev[a[i]] >= 0 or j in used_b:
                continue
            prev[a[i]] = b[j]; used_b.add(j)
    nxt = np.full(len(conf), -1, np.int64)
    has = prev >= 0
    nxt[prev[has]] = np.flatnonzero(has)

    def center(bx):
        return np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])

    def hpt(H, p):
        if H is None:
            return p
        q = H @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    rows_X, rows_lab, rows_meta = [], [], []
    gts_out = []
    for mt in al:
        name, r = mt["reference_name"], int(mt["video_frame_index"])
        if name in exclude or name not in gt or r >= nf:
            continue
        gb, gc, ages = gt[name]
        a = idx[r]
        if len(gb) == 0:
            continue
        # метка и сопоставление по исходной уверенности
        lab = np.zeros(len(a), np.int8)
        if len(a):
            m = np.where(gc[:, None] == cls[a][None, :], iou_mat(gb, xyxy[a]), 0.0)
            used = np.zeros(len(gb), bool)
            for j in np.argsort(-conf[a]):
                col = np.where(used, 0.0, m[:, j]); k = int(col.argmax())
                if col[k] >= 0.5:
                    used[k] = True; lab[j] = 1
        fr_n, fr_mean = len(a), float(conf[a].mean()) if len(a) else 0.0
        for j, di in enumerate(a):
            bx = xyxy[di]; w, h = bx[2] - bx[0], bx[3] - bx[1]; diag = float(np.hypot(w, h)) + 1e-6
            single = [conf[di], np.log1p(w * h), (bx[1] + bx[3]) / 1440.0, (bx[0] + bx[2]) / 2560.0, *np.eye(NCLS)[min(cls[di], NCLS - 1)], fr_n, fr_mean]
            tube, mot = [], []
            for direction, ptr in (("p", prev), ("n", nxt)):
                cs, ious, res_n, res_px, res_vec, scale_res, cnt = [], [], [], [], [], [], 0
                cur = di
                for s in range(K):
                    nb = ptr[cur]
                    if nb < 0:
                        break
                    cnt += 1
                    older, newer = (nb, cur) if direction == "p" else (cur, nb)
                    rr = int(fidx[newer])
                    cs.append(conf[nb]); ious.append(float(iou_mat(xyxy[newer][None], warp_boxes(Hs[rr], xyxy[older][None]))[0, 0]))
                    c0, c1 = center(xyxy[older]), center(xyxy[newer])
                    e = (c1 - c0) - (hpt(Hs[rr], c0) - c0)
                    res_vec.append(e); res_px.append(float(np.hypot(*e))); res_n.append(float(np.hypot(*e)) / diag)
                    wo = warp_boxes(Hs[rr], xyxy[older][None])[0]
                    a_w = max((wo[2] - wo[0]) * (wo[3] - wo[1]), 1.0); a_n = max((xyxy[newer][2] - xyxy[newer][0]) * (xyxy[newer][3] - xyxy[newer][1]), 1.0)
                    scale_res.append(float(np.log(a_n / a_w)))
                    cur = nb
                tube += [cnt, np.mean(cs) if cs else 0, np.min(cs) if cs else 0, np.max(cs) if cs else 0, np.mean(ious) if ious else 0]
                if res_vec:
                    rv = np.array(res_vec); coh = float(np.hypot(*rv.mean(0)) / (np.mean(res_px) + 1e-6))
                    mot += [np.mean(res_n), np.median(res_n), np.mean(res_px), coh, np.mean(np.abs(scale_res)), np.mean(scale_res)]
                else:
                    mot += [-1, -1, -1, -1, -1, 0]
            # остаток DIS внутри рамки на этом кадре
            if r >= 1 and Hs[r] is not None:
                inb = (P[:, 0] >= bx[0]) & (P[:, 0] <= bx[2]) & (P[:, 1] >= bx[1]) & (P[:, 1] <= bx[3])
                if inb.any():
                    pts = P[inb]; ph = np.c_[pts, np.ones(len(pts))] @ Hs[r].T; pred = ph[:, :2] / ph[:, 2:3] - pts
                    rsd = V[r][inb] - pred
                    mot += [float(np.median(np.hypot(rsd[:, 0], rsd[:, 1]))), int(inb.sum())]
                else:
                    mot += [-1, 0]
            else:
                mot += [-1, 0]
            rows_X.append(np.array(single + tube + mot, np.float32)); rows_lab.append(lab[j])
            rows_meta.append((len(gts_out), conf[di]))
        gts_out.append({"name": name, "ages": ages, "n": len(gb),
                        "match": None})
        # для оценки при новых баллах нужно повторное жадное сопоставление: храним IoU-матрицу кадра
        gts_out[-1]["m"] = np.where(gc[:, None] == cls[a][None, :], iou_mat(gb, xyxy[a]), 0.0).astype(np.float16) if len(a) else np.zeros((len(gb), 0), np.float16)
    X = np.stack(rows_X) if rows_X else np.zeros((0, 1), np.float32)
    return vid, X, np.array(rows_lab, np.int8), np.array([m[0] for m in rows_meta], np.int64), gts_out


N_SINGLE = 4 + NCLS + 2
N_TUBE = 10
COLS = {"single": slice(0, N_SINGLE), "tube": slice(N_SINGLE, N_SINGLE + N_TUBE)}


def feature_sets(X):
    s, t, mo = X[:, :N_SINGLE], X[:, N_SINGLE:N_SINGLE + N_TUBE], X[:, N_SINGLE + N_TUBE:]
    tp, mp = t[:, :5], np.c_[mo[:, :6], mo[:, 12:]]   # прошлое: tube «p» и motion «p» + DIS в рамке
    return {"head_single": s, "head_tube": np.c_[s, t], "head_tube_motion": np.c_[s, t, mo],
            "head_tube_motion_causal": np.c_[s, tp, mp]}


def evaluate(V, scores, vids, conf_thr=0.25):
    """Полнота при равном FP (бюджет — исходная уверенность на 0,25) для набора видео."""
    def per_video(v, sc):
        X, lab, fr, gts = V[v]
        found = np.zeros(len(gts), object); fps = []
        for g_i, g in enumerate(gts):
            sel = np.flatnonzero(fr == g_i); m = g["m"].astype(np.float32)
            gs = np.full(g["n"], -np.inf); used = np.zeros(g["n"], bool)
            for jj in np.argsort(-sc[sel]):
                col = np.where(used, 0.0, m[:, jj]) if m.shape[1] else np.zeros(g["n"])
                k = int(col.argmax()) if g["n"] else -1
                if k >= 0 and col[k] >= 0.5:
                    used[k] = True; gs[k] = sc[sel][jj]
                else:
                    fps.append(sc[sel][jj])
            found[g_i] = gs
        return found, np.array(fps)
    out = {}
    res = {name: {v: per_video(v, scores[name][v]) for v in vids} for name in scores}
    budget = sum(int((res["base"][v][1] >= conf_thr).sum()) for v in vids)
    for name in scores:
        if name == "base":
            thr = conf_thr
        else:
            fs = np.sort(np.concatenate([res[name][v][1] for v in vids]))[::-1]
            thr = float(fs[min(budget, len(fs)) - 1])
        out[name] = {"thr": thr, "per_video": {}}
        for v in vids:
            gts = V[v][3]
            f = sum(int((gs >= thr).sum()) for gs in res[name][v][0])
            f0 = sum(int(((gs >= thr) & (g["ages"] == 0)).sum()) for gs, g in zip(res[name][v][0], gts))
            out[name]["per_video"][v] = (f, f0)
    base_res = res["base"]
    n = {v: (sum(g["n"] for g in V[v][3]), sum(int((g["ages"] == 0).sum()) for g in V[v][3])) for v in vids}
    # объекты ступени ПОРОГ: есть кандидат IoU ≥ 0,5 того же класса, но у базы не найден на пороге
    nthr = {}
    for v in vids:
        X, lab, fr, gts = V[v]; cnt = 0
        for g_i, g in enumerate(gts):
            m = g["m"].astype(np.float32)
            if m.shape[1] == 0:
                continue
            sel = np.flatnonzero(fr == g_i); c = X[sel, 0]
            bc = np.where(m >= 0.5, c[None, :], -1.0).max(1)
            cnt += int(((bc >= 0) & (bc < conf_thr)).sum())
        nthr[v] = cnt
    return out, n, nthr, budget


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    seen = {Path(l.strip()).name for l in (PROJECT_ROOT / "training/configs/generated/full_step5_train.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    sets = {"train": sel["train"], "devtest": sel["dev"] + sel["test"], "val7": VAL7}
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}
    tasks = [(v, str(PROJECT_ROOT / "runs/train_cache"), "train", seen) for k in ("train", "devtest") for v in sets[k]]
    tasks += [(v, str(PROJECT_ROOT / "runs/video_rate"), "val", set()) for v in sets["val7"]]
    V = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (vid, X, lab, fr, gts) in enumerate(pool.map(video_features, tasks), 1):
            V[vid] = (X, lab, fr, gts)
            if k % 10 == 0:
                print(f"[FEAT] {k}/{len(tasks)}", flush=True)
    tr = sets["train"]
    Xtr = np.concatenate([V[v][0] for v in tr]); ytr = np.concatenate([V[v][1] for v in tr])
    print(f"[TRAIN] кандидатов {len(ytr)}, верных {int(ytr.sum())}", flush=True)
    names = list(feature_sets(Xtr[:2]).keys())
    clfs = {}
    for name in names:
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
        clf.fit(feature_sets(Xtr)[name], ytr); clfs[name] = clf
        print(f"[HEAD] {name} готова", flush=True)
    res = {"sets": {}}
    rng = np.random.default_rng(0)
    for set_name in ("devtest", "val7"):
        vids = sets[set_name]
        scores = {"base": {v: V[v][0][:, 0] for v in vids}}
        for name in names:
            scores[name] = {v: clfs[name].predict_proba(feature_sets(V[v][0])[name])[:, 1].astype(np.float32) for v in vids}
        out, n, nthr, budget = evaluate(V, scores, vids)
        def share(name, vs, i=0):
            return (sum(out[name]["per_video"][v][i] for v in vs) - sum(out["base"]["per_video"][v][i] for v in vs)) / max(sum(nthr[v] for v in vs), 1)
        r = {"budget": budget, "n_gt": sum(n[v][0] for v in vids), "n_threshold_step": sum(nthr.values()),
             "thresholds": {k: out[k]["thr"] for k in out}, "recall": {}, "net_recovered": {}, "ci95": {}}
        for name in out:
            r["recall"][name] = sum(out[name]["per_video"][v][0] for v in vids) / max(r["n_gt"], 1)
            r["recall"][name + "_age0"] = sum(out[name]["per_video"][v][1] for v in vids) / max(sum(n[v][1] for v in vids), 1)
            r["net_recovered"][name] = share(name, vids)
        pairs = [(nm, "base") for nm in names] + [("head_tube_motion", "head_tube"), ("head_tube", "head_single"), ("head_tube_motion_causal", "head_single")]
        boots = {f"{a}-{b}": [] for a, b in pairs}
        for _ in range(args.boot):
            pick = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
            for a, b in pairs:
                boots[f"{a}-{b}"].append(share(a, pick) - share(b, pick))
        r["diff"] = {k: float(share(k.split("-")[0], vids) - share(k.split("-")[1], vids)) for k in boots}
        r["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        res["sets"][set_name] = r
        print(f"== {set_name}: видео {len(vids)}, эталонов {r['n_gt']}, ступень ПОРОГ {r['n_threshold_step']}, бюджет FP {budget}")
        for name in out:
            print(f"  {name:26s} полнота {r['recall'][name]:.4f} возраст0 {r['recall'][name + '_age0']:.4f} чистый возврат {r['net_recovered'][name]:+.2%}")
        for k in r["diff"]:
            print(f"  {k:40s} {r['diff'][k]:+.2%} [{r['ci95'][k][0]:+.2%}; {r['ci95'][k][1]:+.2%}]")
    imp = {}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
