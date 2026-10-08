"""Т3 трекера на мухе, мишень (карточка fly_assoc_target_v1): потолок сопоставления и узел «нет истории у глаза».

1) Потолок сопоставления при рамках стенда: выход XYWHByteTrack (signatures.npz → outputs: кадр, рамка, номер трека,
   балл, класс) на размеченных кадрах переводится в scalabel как есть (сверка: HOTA стенда) и с номерами треков,
   переназначенными по эталону (рамка ↔ эталонный объект того же класса, IoU ≥ 0,5; без пары — свой номер) — сколько
   HOTA вообще может дать любой модуль сопоставления при тех же рамках. Test 35 и val7.
2) Узнавание после потери (пары «конец старого трека → начало нового», scripts/eval_fly_reid.py:video_pairs): прибавка мухи
   поверх геометрии + внешности + входа глаза — по длине истории нового трека n_first (1–2 и 4–5 кадров). Если прибавка
   растёт с n_first — узел провала узнавания мухой — короткая история глаза на начале трека (как в fly_birth_gate_v1).

Запуск: python scripts/fly_assoc_target.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "fly_assoc_target"
CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def relabel_video(task):
    vid, sig_root, root, split, out_dir = task
    from scripts.oracle_detection_ladder import CATS, iou_mat
    z = np.load(Path(sig_root) / vid / "signatures.npz"); outs = z["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    res = {}
    for mode in ("asis", "oracle"):
        rows, nextid = [], 10 ** 7
        for m, fr in zip(matches, truth):
            f = int(m["video_frame_index"]); o = outs[outs[:, 0] == f]
            gb, gc, gid = [], [], []
            for l in fr.get("labels") or []:
                if l.get("box2d") and l.get("category") in CATS:
                    bb = l["box2d"]; gb.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); gc.append(CATS[l["category"]]); gid.append(l["id"])
            gb = np.array(gb, np.float32).reshape(-1, 4); gc = np.array(gc)
            labels = []
            iou = iou_mat(o[:, 1:5], gb) if len(o) and len(gb) else np.zeros((len(o), 0))
            used = set()
            for i, r in enumerate(o):
                tid = str(int(r[5]))
                if mode == "oracle":
                    tid = None
                    if iou.shape[1]:
                        cand = np.where(gc == int(r[7]), iou[i], 0.0)
                        j = int(cand.argmax())
                        if cand[j] >= 0.5 and j not in used:
                            g = gid[j]; tid = str(2 * 10 ** 8 + (int(g) if str(g).isdigit() else abs(hash(g)) % 10 ** 7)); used.add(j)
                    if tid is None:
                        tid = str(nextid); nextid += 1
                labels.append({"id": tid, "category": CLASS_NAMES[int(r[7])], "score": float(r[6]),
                               "box2d": {"x1": float(r[1]), "y1": float(r[2]), "x2": float(r[3]), "y2": float(r[4])}})
            rows.append({"videoName": vid, "name": fr["name"], "frameIndex": fr["frameIndex"], "labels": labels})
        d = Path(out_dir) / mode / split; d.mkdir(parents=True, exist_ok=True)
        (d / f"{vid}.json").write_text(json.dumps(rows), encoding="utf-8")
    return vid


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    res = {"ceiling": {}}
    rng = np.random.default_rng(1)
    for s in ("test", "val7"):
        tasks = [(v, str(sr), str(rt), sp, str(OUT / "relabel")) for v, sr, rt, sp in S[s]]
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(relabel_video, tasks))
        split = S[s][0][3]; vids = [t[0] for t in tasks]
        agg = {m: evaluate_hota(OUT / "relabel" / m / split, vids, split=split) for m in ("asis", "oracle")}
        per = {m: {v: evaluate_hota(OUT / "relabel" / m / split, [v], split=split)["HOTA"] for v in vids} for m in ("asis", "oracle")}
        d = np.array([per["oracle"][v] - per["asis"][v] for v in vids]); b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        res["ceiling"][s] = {m: {k: agg[m].get(k) for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSw")} for m in agg}
        res["ceiling"][s]["oracle-asis"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
        a, o = res["ceiling"][s]["asis"], res["ceiling"][s]["oracle"]
        print(f"[CEIL] {s}: как есть HOTA {a['HOTA']:.3f} AssA {a['AssA']:.3f} IDSw {a['IDSw']} | номера по эталону HOTA {o['HOTA']:.3f} AssA {o['AssA']:.3f} IDSw {o['IDSw']} | Δ по видео {d.mean():+.3f}", flush=True)
    # 2) узнавание по длине истории нового трека
    from scripts.eval_fly_reid import video_pairs, NAMES, GEOMETRY, APPEAR, HEXIN, FLY
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for s in ("train", "test"):
            data[s] = list(pool.map(video_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
    X = np.concatenate([d["X"] for d in data["train"] if len(d["y"])]); y = np.concatenate([d["y"] for d in data["train"] if len(d["y"])])
    base = [NAMES.index(c) for c in GEOMETRY + APPEAR + HEXIN]; full = base + [NAMES.index(c) for c in FLY]
    mb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[:, base], y)
    mf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[:, full], y)
    Xt = np.concatenate([d["X"] for d in data["test"] if len(d["y"])]); yt = np.concatenate([d["y"] for d in data["test"] if len(d["y"])])
    bl = Xt[:, NAMES.index("b_len")] * 5
    res["reid_by_history"] = {}
    for name, m in (("1-2", bl <= 2), ("3", (bl > 2) & (bl <= 3)), ("4-5", bl >= 4), ("all", np.ones(len(bl), bool))):
        if m.sum() < 50 or not (0 < yt[m].sum() < m.sum()):
            continue
        pb = mb.predict_proba(Xt[m][:, base])[:, 1]; pf = mf.predict_proba(Xt[m][:, full])[:, 1]
        idx = np.flatnonzero(m); bs = []
        for _ in range(1000):
            k = rng.integers(0, len(idx), len(idx))
            if 0 < yt[m][k].sum() < len(k):
                bs.append(average_precision_score(yt[m][k], pf[k]) - average_precision_score(yt[m][k], pb[k]))
        r = {"n": int(m.sum()), "pos": float(yt[m].mean()), "AP_base": float(average_precision_score(yt[m], pb)), "AP_fly": float(average_precision_score(yt[m], pf)),
             "ci95_pairs": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        r["gain"] = r["AP_fly"] - r["AP_base"]; res["reid_by_history"][name] = r
        print(f"[REID] n_first {name}: пар {r['n']} (положит. {r['pos']:.2f}) AP без мухи {r['AP_base']:.4f} с мухой {r['AP_fly']:.4f} прибавка {r['gain']:+.4f} [{r['ci95_pairs'][0]:+.4f}; {r['ci95_pairs'][1]:+.4f}] (бутстрэп по парам)", flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
