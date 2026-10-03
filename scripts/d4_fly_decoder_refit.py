"""Д4: откуда ошибка движения мухи — от декодера Sintel или от самого представления (карточка fly_decoder_refit_v1).

Данные: val7, сохранённая активность FlyVis (runs/flyvis_activity/<id>/flyvis_tiled_activity.npz: кадр × 6 плиток ×
34 выходных типа × 721 колонка, координаты колонок), исходный декодированный поток мухи (runs/video_rate/<id>/motion/
flyvis_tiled.npz) и DIS (dis_quarter.npz). Поля движения:
  zero      — нулевое движение;
  dis       — DIS;
  fly_orig  — штатный декодер FlyVis (обучен на Sintel);
  fly_refit — гребневая регрессия активности колонки и 6 соседей (34 × 2 признака, текущий и прошлый кадр) → DIS в этой точке;
              обучение с исключением видео по одному (декодер для видео v не видел v);
  inp_refit — контроль «вход без мухи»: та же регрессия на яркостях (размытие 13 px, как фасетка) в колонке и соседях
              за текущий и прошлый кадр и их попарных произведениях (корреляционные признаки, как у детектора Райхардта).
Метрика (как в evaluate_motion_sources): для эталонных объектов, присутствующих на двух соседних размеченных кадрах,
рамка переносится полем по сырым кадрам (scripts.evaluate_flyvis_qmatch_gate.propagate_box); ошибка — расстояние
между центром перенесённой и эталонной рамки, px; медиана и среднее; ночь/день по атрибутам.

Правило (записано до прогона): fly_refit снижает медианную ошибку ≥ 50 % относительно fly_orig — изъян в считывании;
иначе — в представлении. fly_refit лучше inp_refit (ДИ по видео > 0) — муха несёт движение сверх своего входа.

Запуск: python scripts/d4_fly_decoder_refit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117"]  # только у этих четырёх сохранён поток мухи (flyvis_tiled.npz)
OUT = PROJECT_ROOT / "runs" / "d4_decoder_refit"
CATS = {"pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
K_NB = 6
TRAIN_FRACTION = 0.02


def neighbours(points):
    from scipy.spatial import cKDTree
    tiles = points.reshape(6, 721, 2)
    out = np.zeros((6, 721, K_NB), np.int64)
    for t in range(6):
        _, idx = cKDTree(tiles[t]).query(tiles[t], k=K_NB + 1)
        out[t] = idx[:, 1:]
    return out


def dis_at(points_dis, vectors_dis, pts):
    """DIS в точках колонок: ближайший узел сетки DIS (шаг 32 px)."""
    from scipy.spatial import cKDTree
    _, idx = cKDTree(points_dis).query(pts, k=1)
    return vectors_dis[:, idx]  # (F, N, 2)


def fly_features(act, base, nb, f):
    """(N=4326, 136): активность колонки и среднее по соседям, текущий и прошлый кадр, минус базовая."""
    feats = []
    for g in (f, max(f - 1, 0)):
        a = act[g].astype(np.float32) - base.astype(np.float32)   # (6, 34, 721)
        a = a.transpose(0, 2, 1)                                  # (6, 721, 34)
        nbm = np.stack([a[t][nb[t]].mean(1) for t in range(6)])   # (6, 721, 34)
        feats += [a.reshape(-1, 34), nbm.reshape(-1, 34)]
    return np.concatenate(feats, 1)


def input_features(blur, pts_int, nb_flat, f):
    """Контроль: яркости колонки и соседей (кадры f и f−1) и их попарные произведения."""
    c0, p0 = blur[f], blur[max(f - 1, 0)]                          # (N,) — уже яркости в точках колонок
    cn = c0[nb_flat]; pn = p0[nb_flat]                            # (N, 6)
    return np.concatenate([c0[:, None], p0[:, None], cn, pn, c0[:, None] * pn, p0[:, None] * cn, (c0 - p0)[:, None] * (cn - pn)], 1)


def main() -> int:
    import cv2
    from sklearn.linear_model import Ridge
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box

    OUT.mkdir(parents=True, exist_ok=True)
    attrs = load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_val.json"))
    cond = {}
    for c in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, VAL7, c):
            cond.setdefault(v, c)
    D = {}
    rng = np.random.default_rng(0)
    for v in VAL7:
        z = np.load(PROJECT_ROOT / "runs/flyvis_activity" / v / "flyvis_tiled_activity.npz")
        act, base, pts = z["frames"], z["baseline"], z["points"].astype(np.float32)
        fo = np.load(PROJECT_ROOT / "runs/video_rate" / v / "motion/flyvis_tiled.npz")
        dq = np.load(PROJECT_ROOT / "runs/video_rate" / v / "motion/dis_quarter.npz")
        nb = neighbours(pts)
        nb_flat = (nb + (np.arange(6) * 721)[:, None, None]).reshape(-1, K_NB)
        target = dis_at(dq["points"], dq["vectors"], pts)
        cap = cv2.VideoCapture(str(PROJECT_ROOT / "data/bdd100k/videos/val" / f"{v}.mov")); blur = []
        pts_int = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.blur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, (13, 13))
            if pts_int is None:
                h, w = g.shape
                pts_int = np.c_[np.clip(pts[:, 0].round(), 0, w - 1), np.clip(pts[:, 1].round(), 0, h - 1)].astype(int)
            blur.append(g[pts_int[:, 1], pts_int[:, 0]])   # только яркости в точках колонок
        cap.release()
        blur = np.stack(blur)                               # (F, N)
        nf = min(len(act), len(blur), len(target))
        pts_int = np.c_[np.arange(len(pts)), np.zeros(len(pts), int)]  # индексация уже выбранных точек
        D[v] = dict(act=act, base=base, pts=pts, nb=nb, nb_flat=nb_flat, target=target, blur=blur, pts_int=pts_int, nf=nf,
                    fly_orig=fo["vectors"], dis_pts=dq["points"], dis_vec=dq["vectors"])
        print(f"[LOAD] {v} кадров {nf}", flush=True)

    def sample(v, kind):
        d = D[v]; Xs, Ys = [], []
        frames = rng.choice(np.arange(1, d["nf"]), max(1, int(d["nf"] * 0.15)), replace=False)
        for f in frames:
            X = fly_features(d["act"], d["base"], d["nb"], f) if kind == "fly" else input_features(d["blur"], d["pts_int"], d["nb_flat"], f)
            y = d["target"][f]
            ok = np.isfinite(y).all(1) & np.isfinite(X).all(1)
            pick = np.flatnonzero(ok); pick = rng.choice(pick, max(1, int(len(pick) * TRAIN_FRACTION * 7)), replace=False)
            Xs.append(X[pick]); Ys.append(y[pick])
        return np.concatenate(Xs), np.concatenate(Ys)

    samples = {kind: {v: sample(v, kind) for v in VAL7} for kind in ("fly", "inp")}
    fields = {}
    for v in VAL7:
        d = D[v]
        fields[v] = {"zero": np.zeros_like(d["fly_orig"]), "dis": None, "fly_orig": d["fly_orig"]}
        for kind in ("fly", "inp"):
            X = np.concatenate([samples[kind][u][0] for u in VAL7 if u != v]); Y = np.concatenate([samples[kind][u][1] for u in VAL7 if u != v])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            model = Ridge(alpha=10.0).fit((X - mu) / sd, Y)
            pred = np.zeros((d["nf"], len(d["pts"]), 2), np.float32)
            for f in range(1, d["nf"]):
                Xf = fly_features(d["act"], d["base"], d["nb"], f) if kind == "fly" else input_features(d["blur"], d["pts_int"], d["nb_flat"], f)
                pred[f] = model.predict((np.nan_to_num(Xf) - mu) / sd)
            fields[v][f"{kind}_refit"] = pred
        print(f"[FIT] {v}", flush=True)

    # метрика: перенос рамки между соседними размеченными кадрами
    res = {}
    for v in VAL7:
        d = D[v]
        al = json.loads((PROJECT_ROOT / "runs/video_rate" / v / "alignment.json").read_text(encoding="utf-8"))["matches"]
        truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/val" / f"{v}.json").read_text(encoding="utf-8"))
        boxes = []
        for m, fr in zip(al, truth):
            boxes.append((int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                                        for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in CATS}))
        errs = {k: [] for k in ("zero", "dis", "fly_orig", "fly_refit", "inp_refit")}
        for (r0, b0), (r1, b1) in zip(boxes[:-1], boxes[1:]):
            if r1 <= r0 or r1 >= d["nf"]:
                continue
            for tid, box in b0.items():
                if tid not in b1:
                    continue
                gt_c = (b1[tid][:2] + b1[tid][2:]) / 2
                for k in errs:
                    if k == "dis":
                        pred = propagate_box(box, d["dis_pts"], d["dis_vec"], r0, r1)[0]
                    else:
                        pred = propagate_box(box, d["pts"], fields[v][k], r0, r1)[0]
                    errs[k].append(float(np.linalg.norm((pred[:2] + pred[2:]) / 2 - gt_c)))
        res[v] = {k: {"median": float(np.median(e)), "mean": float(np.mean(e)), "n": len(e)} for k, e in errs.items()}
        print(f"[ERR] {v} ({cond.get(v, '—')}): " + " | ".join(f"{k} {res[v][k]['median']:.2f}" for k in errs), flush=True)

    groups = {"all": VAL7, "night": [v for v in VAL7 if cond.get(v) == "night"], "day": [v for v in VAL7 if cond.get(v) == "normal"]}
    summary = {}
    boot_rng = np.random.default_rng(1)
    for g, vs in groups.items():
        if not vs:
            continue
        summary[g] = {k: float(np.mean([res[v][k]["median"] for v in vs])) for k in res[vs[0]]}
        for a, b in (("fly_refit", "fly_orig"), ("fly_refit", "inp_refit"), ("fly_refit", "dis")):
            dlt = np.array([res[v][a]["median"] - res[v][b]["median"] for v in vs])
            bs = dlt[boot_rng.integers(0, len(dlt), (5000, len(dlt)))].mean(1)
            summary[g][f"{a}-{b}"] = {"mean": float(dlt.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
        summary[g]["reduction_refit_vs_orig"] = 1 - summary[g]["fly_refit"] / summary[g]["fly_orig"]
    out = {"per_video": res, "summary": summary, "cond": cond}
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, s in summary.items():
        print(f"== {g}: zero {s['zero']:.2f} dis {s['dis']:.2f} fly_orig {s['fly_orig']:.2f} fly_refit {s['fly_refit']:.2f} inp_refit {s['inp_refit']:.2f} | снижение refit/orig {s['reduction_refit_vs_orig']:.1%}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
