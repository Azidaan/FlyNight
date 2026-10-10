"""Буфер трекера 30 против 150 кадров без мухи, та же сшивка MASA (карточка buffer_extend_v1, шаг А1 после аудита).

lost_link_audit_v1: после MASA на dev остаётся 401 разрыв > 30 кадров, запас связей +1,33 HOTA. Здесь — дешёвое действие
без мухи: тот же стенд, буфер потерянного трека 150 кадров вместо 30 (scripts/build_fly_signatures.py --track-buffer 150 →
runs/fly_signatures_buf150; эмбеддинги — runs/dino_embeddings_buf150, runs/masa_embeddings_buf150), затем та же сшивка
«база + MASA» (окно 150; 5, порог 0,6; голова пары — обученная на парах train 130 с буфером 30, как в stitch_eval_v2 —
признаки пары от буфера не зависят, оговорка: жизни dev при буфере 150 могут иметь другое распределение).
Сравнение на dev 35: выход после MASA при буфере 150 против выхода после MASA при буфере 30 (runs/chain_filter_rank/repro/dev);
справочно — без сшивки (буфер 150 против 30: runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev) и число оставшихся длинных
разрывов. Критерий (до прогона): HOTA (после MASA) буфер 150 − буфер 30 по видео > 0, ДИ > 0 (бутстрэп 5000).

Запуск: python scripts/buffer_extend_eval.py
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

SIG = PROJECT_ROOT / "runs/fly_signatures_buf150"
DINO, MASA = PROJECT_ROOT / "runs/dino_embeddings_buf150", PROJECT_ROOT / "runs/masa_embeddings_buf150"
OUT = PROJECT_ROOT / "runs/buffer_extend"
B30_MASA = PROJECT_ROOT / "runs/chain_filter_rank/repro/dev"
B30_NONE = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev"
KEYS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def per_video_hota(task):
    from evaluation.hota_eval import evaluate_hota
    d, vid, split = task
    m = evaluate_hota(Path(d), [vid], split=split)
    return vid, {k: m.get(k) for k in KEYS}


def long_breaks(task):
    from scripts.chain_filter_head import classify_ids
    d, vid, root, split = task
    res = json.loads((Path(d) / f"{vid}.json").read_text(encoding="utf-8"))
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    seq = {}
    for k, (fr, tg) in enumerate(zip(res, classify_ids(res, gt))):
        for l, (t, g) in zip(fr.get("labels") or [], tg):
            if t == "tp":
                seq.setdefault(g, []).append((k, vf[k], str(l["id"])))
    n = sum(1 for s in seq.values() for a, b in zip(sorted(s), sorted(s)[1:]) if a[2] != b[2] and b[1] - a[1] > 30)
    m = sum(1 for s in seq.values() for a, b in zip(sorted(s), sorted(s)[1:]) if a[2] != b[2])
    return vid, n, m


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"] if (SIG / v / "signatures.npz").exists() and (MASA / f"{v}.npz").exists()]
    vids = [v for v, _, _ in dev]; split = dev[0][2]; root = dev[0][1]
    print(f"[PLAN] dev с буфером 150: {len(vids)} видео", flush=True)
    with ProcessPoolExecutor(max_workers=4, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    with ProcessPoolExecutor(max_workers=4, initializer=V._set_dirs, initargs=(150, 5.0, str(DINO), str(MASA))) as pool:
        pairs = list(pool.map(V.video_pairs, [(v, str(SIG), root, split) for v in vids]))
    for thr, tag in ((0.6, "masa"), (2.0, "none")):
        tasks = [(d["video_id"], str(SIG), root, split, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), thr, str(OUT / f"b150_{tag}")) for d in pairs]
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(stitch_video, tasks))
    print("[STITCH] буфер 150: сшитый выход и без сшивки готовы", flush=True)
    dirs = {"b30_masa": B30_MASA, "b150_masa": OUT / "b150_masa", "b30_none": B30_NONE, "b150_none": OUT / "b150_none"}
    per = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, d in dirs.items():
            per[name] = dict(pool.map(per_video_hota, [(str(d), v, split) for v in vids]))
        lb = {name: {v: (n, m) for v, n, m in pool.map(long_breaks, [(str(dirs[name]), v, root, split) for v in vids])} for name in ("b30_masa", "b150_masa")}
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    night = list(filter_videos_by_subset(attrs, vids, "night"))
    rep = {"videos": len(vids), "pooled": {}, "diff": {}, "breaks": {k: {"long": int(sum(x[0] for x in v.values())), "all": int(sum(x[1] for x in v.values()))} for k, v in lb.items()}}
    rng = np.random.default_rng(1)
    for g, gv in (("all", vids), ("night", night)):
        for name, d in dirs.items():
            m = evaluate_hota(d, gv, split=split); rep["pooled"][f"{name}|{g}"] = {k: m.get(k) for k in KEYS}
        for a_, b_ in (("b150_masa", "b30_masa"), ("b150_none", "b30_none"), ("b30_masa", "b30_none"), ("b150_masa", "b150_none")):
            for met in ("HOTA", "AssA", "DetA"):
                dd = np.array([per[a_][v][met] - per[b_][v][met] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                rep["diff"][f"{a_}-{b_}|{g}|{met}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "n": len(dd)}
    d = rep["diff"]["b150_masa-b30_masa|all|HOTA"]; rep["criterion_met"] = bool(d["mean_per_video"] > 0 and d["ci95"][0] > 0)
    for g in ("all", "night"):
        print(f"[HOTA] {g}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{g}']['HOTA']:.3f} (AssA {rep['pooled'][f'{n}|{g}']['AssA']:.2f}, IDSw {rep['pooled'][f'{n}|{g}']['IDSw']:.0f})" for n in dirs), flush=True)
        for k, x in rep["diff"].items():
            if f"|{g}|HOTA" in k:
                print(f"[DIFF] {k}: {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[BREAKS] {rep['breaks']}", flush=True)
    print(f"[CRITERION] {rep['criterion_met']}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
