"""Когда движению мухи можно доверять (карточка fly_motion_trust_v1; шаг 1).

Для каждого перехода между соседними размеченными кадрами (dev 35, train_cache) и каждого эталонного объекта,
присутствующего на обоих кадрах: рамка эталона в начале переносится полем мухи + lcn24 (усиление 3), полем DIS,
постоянной скоростью (по эталону двух предыдущих размеченных кадров — заменитель прогноза Калмана) и нулём.
Записываются: ошибка центра в конце (px) для каждого источника; уверенность поля мухи (средний вес propagate_box —
та же величина, что смешивает прогноз в motion_scheme_search.py); размер объекта; число колонок мухи внутри рамки;
расхождение мухи с постоянной скоростью; ошибка мухи за первый сырой кадр (без накопления) против DIS за тот же кадр.
Сводка: различает ли уверенность полезные и вредные поправки (Спирмен, AUC «муха лучше постоянной скорости»),
по размеру и числу колонок; потолок идеального выбора «муха / постоянная скорость» на уровне переходов.

Запуск: python scripts/fly_trust_diagnostic.py [--set val7]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "fly_motion_trust"
CATS = {"pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}


def main() -> int:
    global OUT
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    from scripts.run_bdd_video_rate_experiment import load_motion_cache
    from tracking.flyvis_tracker import MotionField
    OUT.mkdir(parents=True, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=("dev", "val7"), default="dev"); a = ap.parse_args()
    if a.set == "dev":
        tc = PROJECT_ROOT / "runs/train_cache"; split = "train"
        vids = json.loads((tc / "selection.json").read_text(encoding="utf-8"))["videos"]["dev"]
    else:   # проверка поля мухи на val7 (08.10)
        from scripts.build_fly_lcn_motion import VAL7
        tc = PROJECT_ROOT / "runs/video_rate"; split = "val"; vids = VAL7
    if a.set != "dev":
        OUT = OUT / a.set; OUT.mkdir(parents=True, exist_ok=True)
    rows = []; t0 = time.time()
    for i, v in enumerate(vids):
        fpts, fvec = load_motion_cache(tc / v / "motion/flyvis_tiled_lcn24.npz", 3.0)
        dpts, dvec = load_motion_cache(tc / v / "motion/dis_quarter.npz", 1.0)
        al = json.loads((tc / v / "alignment.json").read_text(encoding="utf-8"))["matches"]
        truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{v}.json").read_text(encoding="utf-8"))
        boxes = [(int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                                for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in CATS}) for m, fr in zip(al, truth)]
        nf = min(len(fvec), len(dvec))
        for k in range(1, len(boxes) - 1):
            (rp, bp), (r0, b0), (r1, b1) = boxes[k - 1], boxes[k], boxes[k + 1]
            if not (rp < r0 < r1 < nf):
                continue
            for tid, box in b0.items():
                if tid not in b1 or not (box[2] - box[0] > 1 and box[3] - box[1] > 1):
                    continue
                c0 = (box[:2] + box[2:]) / 2; gt = (b1[tid][:2] + b1[tid][2:]) / 2
                pf, wf = propagate_box(box, fpts, fvec, r0, r1)
                pd, wd = propagate_box(box, dpts, dvec, r0, r1)
                cf = (pf[:2] + pf[2:]) / 2; cd = (pd[:2] + pd[2:]) / 2
                if tid in bp:
                    vel = (c0 - (bp[tid][:2] + bp[tid][2:]) / 2) / (r0 - rp); ccv = c0 + vel * (r1 - r0)
                else:
                    ccv = c0
                inside = int(np.all((fpts >= box[:2]) & (fpts <= box[2:]), axis=1).sum())
                e1f = MotionField(fpts, fvec[r0 + 1]).estimate_box(box, context_scale=2.0, min_samples=3)
                e1d = MotionField(dpts, dvec[r0 + 1]).estimate_box(box, context_scale=2.0, min_samples=3)
                rows.append([np.linalg.norm(cf - gt), np.linalg.norm(cd - gt), np.linalg.norm(ccv - gt), np.linalg.norm(c0 - gt),
                             wf, wd, float(np.sqrt((box[2] - box[0]) * (box[3] - box[1]))), inside, np.linalg.norm(cf - ccv),
                             float(np.linalg.norm(e1f.vector_xy - e1d.vector_xy)), float(e1f.confidence), i, int(tid in bp)])
        print(f"[TRUST] {i + 1}/{len(vids)} {v}: переходов-объектов {len(rows)} | {time.time() - t0:.0f} с", flush=True)
    R = np.array(rows, np.float64)
    np.save(OUT / "rows.npy", R)
    ef, ed, ecv, ez, wf, wd, size, inside, dis_cv, e1, c1, vid, hascv = R.T
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    better = ef < ecv
    res = {"n": int(len(R)), "median_err": {"fly": float(np.median(ef)), "dis": float(np.median(ed)), "const_velocity": float(np.median(ecv)), "zero": float(np.median(ez))},
           "share_fly_better_than_cv": float(better.mean()), "share_dis_better_than_cv": float((ed < ecv).mean()),
           "spearman_conf_vs_flyerr": float(spearmanr(wf, ef).correlation), "auc_conf_fly_better_cv": float(roc_auc_score(better, wf)),
           "spearman_conf1_vs_step1err": float(spearmanr(c1, e1).correlation),
           "auc_disagreement_fly_better_cv": float(roc_auc_score(better, -dis_cv)),
           "oracle_choice_median_err": float(np.median(np.minimum(ef, ecv))),
           "mean_err": {"fly": float(ef.mean()), "dis": float(ed.mean()), "cv": float(ecv.mean()), "oracle_fly_cv": float(np.minimum(ef, ecv).mean())}}
    bins = {}
    for name, x, edges in (("size", size, [0, 32, 64, 128, 1e9]), ("inside_cols", inside, [-1, 0, 2, 5, 1e9])):
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x > a) & (x <= b)
            if m.sum() > 50:
                bins[f"{name}_{a:g}-{b:g}"] = {"n": int(m.sum()), "fly": float(np.median(ef[m])), "dis": float(np.median(ed[m])), "cv": float(np.median(ecv[m])),
                                               "fly_better_cv": float(better[m].mean()), "auc_conf": float(roc_auc_score(better[m], wf[m])) if 0 < better[m].mean() < 1 else None, "mean_conf_fly": float(wf[m].mean())}
    res["bins"] = bins
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
