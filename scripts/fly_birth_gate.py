"""Т2 трекера на мухе, гейт (карточка fly_birth_gate_v1): отличает ли отпечаток мухи настоящее рождение трека от ложного.

Мишень проверена до стройки (правило 6.1): потолок «идеальное рождение молодых объектов» +0,21 HOTA на test
(detection_oracle_ladder_v1); муха в голове q_birth по тайловому считыванию ничего не добавляла (flyvis_event_features_v1).
Здесь — отпечаток фовеального глаза мухи на начале жизни трека (runs/fly_signatures: first_fly 34×19, first_hexin 721 вход,
first_thumb/hist — внешность), как для «объект ещё здесь?» (fly_tracker_lifecycle_v1).
Событие — начало жизни глаза трека; метка «настоящее рождение» — на первом размеченном кадре жизни рамка трека совпадает
с эталонным объектом того же класса (IoU ≥ 0,5). Головы (логистическая регрессия): geometry, geometry_fly, geometry_hexin,
geometry_hexin_fly, geometry_appearance. Обучение — train 130, проверка — test 35 и val7; AP и бутстрэп по видео.

Запуск: python scripts/fly_birth_gate.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fly_lifecycle_alive import ap, build_X, sets  # noqa: E402
from scripts.oracle_detection_ladder import CATS, iou_mat  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_birth_gate"
HEADS = {"geometry": ["geo"], "geometry_fly": ["geo", "fly"], "geometry_hexin": ["geo", "hexin"], "geometry_hexin_fly": ["geo", "hexin", "fly"], "geometry_appearance": ["geo", "app"]}


def video_births(task):
    vid, sig_root, root, split = task
    z = np.load(Path(sig_root) / vid / "signatures.npz")
    outs = z["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    lab_r = np.array([int(m["video_frame_index"]) for m in matches])
    gt = []
    for fr in truth:
        b, c = [], []
        for l in fr.get("labels") or []:
            if l.get("box2d") and l.get("category") in CATS:
                bb = l["box2d"]; b.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); c.append(CATS[l["category"]])
        gt.append((np.array(b, np.float32).reshape(-1, 4), np.array(c)))
    rows = []
    for k in range(len(z["track_id"])):
        tid, st, en = int(z["track_id"][k]), int(z["start"][k]), int(z["end"][k])
        inlife = np.flatnonzero((lab_r >= st) & (lab_r <= en))
        if not len(inlife):
            continue
        i0 = inlife[0]; r0 = lab_r[i0]
        o = outs[(outs[:, 0] == r0) & (outs[:, 5] == tid)]
        if not len(o):
            continue
        gb, gc = gt[i0]
        true = 0
        if len(gb):
            m = np.where(gc == int(z["cls"][k]), iou_mat(o[:1, 1:5], gb)[0], 0.0)
            true = int(m.max() >= 0.5)
        fb = z["first_box"][k].astype(np.float32); w, h = max(fb[2] - fb[0], 1.0), max(fb[3] - fb[1], 1.0)
        conf = float(o[0, 6]) if o.shape[1] > 6 else 0.0
        geo = np.r_[fb / np.array([1280, 720, 1280, 720]), np.log1p(w * h), w / max(h, 1), np.eye(8)[int(z["cls"][k]) % 8], float(z["n_first"][k]), conf, (r0 - st) / 30.0]
        rows.append(dict(vid=vid, tid=tid, true=true, geo=geo.astype(np.float32), fly=z["first_fly"][k].astype(np.float32).ravel(),
                         hexin=z["first_hexin"][k].astype(np.float32), app=np.r_[z["first_thumb"][k], z["first_hist"][k]].astype(np.float32)))
    return vid, rows


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    ev = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for s in ("train", "test", "val7"):
            ev[s] = dict(pool.map(video_births, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
    tr = [r for v in ev["train"] for r in ev["train"][v]]; y = np.array([r["true"] for r in tr])
    res = {"n_train": len(tr), "true_share_train": float(y.mean()), "AP": {}, "diff": {}}
    print(f"рождений в train {len(tr)}, доля настоящих {y.mean():.3f}", flush=True)
    models = {h: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(build_X(tr, p), y) for h, p in HEADS.items()}
    rng = np.random.default_rng(0)
    for s in ("test", "val7"):
        vids = [v for v in ev[s] if ev[s][v]]
        per = {h: {v: (np.array([r["true"] for r in ev[s][v]]), models[h].predict_proba(build_X(ev[s][v], HEADS[h]))[:, 1]) for v in vids} for h in HEADS}
        for h in HEADS:
            yy = np.concatenate([per[h][v][0] for v in vids]); pp = np.concatenate([per[h][v][1] for v in vids])
            res["AP"][f"{h}|{s}"] = {"AP": ap(yy, pp), "n": int(len(yy)), "true_share": float(yy.mean())}
        for a, b in (("geometry_fly", "geometry"), ("geometry_hexin", "geometry"), ("geometry_hexin_fly", "geometry_hexin"), ("geometry_appearance", "geometry"), ("geometry_fly", "geometry_appearance")):
            bs = []
            for _ in range(2000):
                pick = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
                ya = np.concatenate([per[a][v][0] for v in pick])
                bs.append(ap(ya, np.concatenate([per[a][v][1] for v in pick])) - ap(ya, np.concatenate([per[b][v][1] for v in pick])))
            bs = np.array([x for x in bs if np.isfinite(x)])
            res["diff"][f"{a}-{b}|{s}"] = {"mean": res["AP"][f"{a}|{s}"]["AP"] - res["AP"][f"{b}|{s}"]["AP"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    for k, v in res["AP"].items():
        print(f"  AP {k:28s} {v['AP']:.4f} (n {v['n']}, настоящих {v['true_share']:.2f})", flush=True)
    for k, v in res["diff"].items():
        print(f"  Δ {k:36s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}; {v['ci95'][1]:+.4f}]", flush=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
