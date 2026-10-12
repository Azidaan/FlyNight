"""З13 (карточка fly_object_readout_z13_v1): оценка вариантов на dev 35 после сшивки MASA.

Для каждого варианта (B, C, D) — выход трекера со считывателем (scripts/build_fly_signatures.py --z13 runs/z13/<arm>/model.pt →
runs/fly_signatures_z13_<arm>), эмбеддинги DINO / MASA (runs/dino_embeddings_z13_<arm>, runs/masa_embeddings_z13_<arm>),
та же сшивка «база + MASA» (окно 150; 5, порог 0,6; голова пары на train 130 стенда — как scripts/o2_masa_residual_eval.py).
A — стенд с MASA (runs/chain_filter_rank/repro/dev, 61,655). Критерии (до прогона, карточка):
  польза реализации — D − A: общий прирост ≥ +0,10 и ДИ средней разности по видео > 0;
  добавочная польза мухи — D − B: средняя по видео > 0, ДИ > 0;  польза обучения ядра — D − C: средняя по видео > 0, ДИ > 0.
Выводы не заменяют друг друга. Выход — runs/z13/eval/report.json.

Запуск: python scripts/z13_eval.py [--arms B C D]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402
from scripts.buffer_extend_eval import KEYS, per_video_hota  # noqa: E402

OUT = PROJECT_ROOT / "runs/z13/eval"
STAND_MASA = PROJECT_ROOT / "runs/chain_filter_rank/repro/dev"
STAND_NONE = PROJECT_ROOT / "runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev"
PRACTICAL_MIN = 0.10


def dirs(arm: str):
    return PROJECT_ROOT / f"runs/fly_signatures_z13_{arm}", PROJECT_ROOT / f"runs/dino_embeddings_z13_{arm}", PROJECT_ROOT / f"runs/masa_embeddings_z13_{arm}"


def paired(per, a, b, vids, rng):
    d = np.array([per[a][v]["HOTA"] - per[b][v]["HOTA"] for v in vids]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default=["B", "C", "D"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    vids = [v for v, _, _ in dev]; split = dev[0][2]; root = dev[0][1]
    for arm in a.arms:
        sig, _, masa = dirs(arm)
        miss = [v for v in vids if not (sig / v / "signatures.npz").exists() or not (masa / f"{v}.npz").exists()]
        if miss:
            raise SystemExit(f"[STOP] вариант {arm}: нет сигнатур или MASA для {len(miss)} видео dev — оценка только на полном dev 35")
    with ProcessPoolExecutor(max_workers=4, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    out_dirs = {"A_masa": STAND_MASA, "A_none": STAND_NONE}
    events = {}
    for arm in a.arms:
        sig, dino, masa = dirs(arm)
        with ProcessPoolExecutor(max_workers=4, initializer=V._set_dirs, initargs=(150, 5.0, str(dino), str(masa))) as pool:
            pairs = list(pool.map(V.video_pairs, [(v, str(sig), root, split) for v in vids]))
        for thr, tag in ((0.6, "masa"), (2.0, "none")):
            tasks = [(d["video_id"], str(sig), root, split, d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), thr, str(OUT / f"{arm}_{tag}")) for d in pairs]
            with ProcessPoolExecutor(max_workers=4) as pool:
                list(pool.map(stitch_video, tasks))
            out_dirs[f"{arm}_{tag}"] = OUT / f"{arm}_{tag}"
        ev = {}
        for v in vids:
            for k, x in json.loads((sig / v / "signatures.json").read_text(encoding="utf-8")).get("z13_events", {}).items():
                ev[k] = ev.get(k, 0) + int(x)
        events[arm] = ev
        print(f"[STITCH] {arm}: готово; события {ev}", flush=True)
    per = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for name, d in out_dirs.items():
            per[name] = dict(pool.map(per_video_hota, [(str(d), v, split) for v in vids]))
    night = list(filter_videos_by_subset(load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json")), vids, "night"))
    rng = np.random.default_rng(1)
    rep = {"videos": len(vids), "events": events, "pooled": {}, "diff": {}}
    for g, gv in (("all", vids), ("night", night)):
        for name, d in out_dirs.items():
            m = evaluate_hota(d, gv, split=split); rep["pooled"][f"{name}|{g}"] = {k: m.get(k) for k in KEYS}
        pairs_ = [("D_masa", "A_masa"), ("D_masa", "B_masa"), ("D_masa", "C_masa"), ("B_masa", "A_masa"), ("C_masa", "A_masa"), ("D_none", "A_none")]
        for x, y_ in pairs_:
            if x in per and y_ in per:
                r = paired(per, x, y_, gv, rng); r["pooled_gain"] = rep["pooled"][f"{x}|{g}"]["HOTA"] - rep["pooled"][f"{y_}|{g}"]["HOTA"]
                rep["diff"][f"{x}-{y_}|{g}"] = r
    D = rep["diff"]
    rep["criteria"] = {
        "implementation_D_vs_A": bool("D_masa-A_masa|all" in D and D["D_masa-A_masa|all"]["pooled_gain"] >= PRACTICAL_MIN and D["D_masa-A_masa|all"]["ci95"][0] > 0),
        "fly_added_D_vs_B": bool("D_masa-B_masa|all" in D and D["D_masa-B_masa|all"]["mean_per_video"] > 0 and D["D_masa-B_masa|all"]["ci95"][0] > 0),
        "core_training_D_vs_C": bool("D_masa-C_masa|all" in D and D["D_masa-C_masa|all"]["mean_per_video"] > 0 and D["D_masa-C_masa|all"]["ci95"][0] > 0),
    }
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ("all", "night"):
        print(f"[HOTA] {g}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{g}']['HOTA']:.3f}" for n in out_dirs), flush=True)
        for k, x in D.items():
            if k.endswith(f"|{g}"):
                print(f"[DIFF] {k}: общий {x['pooled_gain']:+.3f}; по видео {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[CRITERIA] {rep['criteria']}\n[OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
