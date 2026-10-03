"""Гейт рамок: сколько HOTA дают сглаживание траектории и выбор рамки на выходе трекера (карточка box_smoothing_v1).

Основание: лестница оракулов (detection_oracle_ladder_v1) — точные рамки +5,4 HOTA, крупнейший рычаг.
Выход трекера — апостериорная оценка Калмана (STrack.tlwh = mean). Варианты на сохранённых выходах 30 Гц
(runs/line_b_combined/outputs/<трекер>/<id>.npz; кадр, x1, y1, x2, y2, id, балл, класс):
  base        — как есть;
  det         — рамка заменена рамкой сопоставленной детекции этого кадра (тот же класс, IoU ≥ 0,6, максимум IoU);
  ma5/9/15    — центрированное скользящее среднее по треку (центр и размер), окно 5/9/15 кадров (офлайн);
  lag3/lag6   — окно [t − L, t + L] (задержка L кадров: 0,1/0,2 с при 30 Гц);
  interp      — линейное заполнение пропусков трека ≤ 20 кадров (как GSI без сглаживания);
  ma9_interp  — ma9 + interp.
Трекеры: stand и cascade_deferred (без сшивки). Наборы: test 35, val7; HOTA TrackEval; бутстреп по видео.

Запуск: python scripts/gate_box_smoothing.py --workers 10
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

OUT = PROJECT_ROOT / "runs" / "gate_box"
SRC = PROJECT_ROOT / "runs" / "line_b_combined" / "outputs"
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
TRACKERS = ["stand", "cascade_deferred"]
VARIANTS = ["base", "det", "ma5", "ma9", "ma15", "lag3", "lag6", "interp", "ma9_interp"]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")
MAX_GAP = 20


def to_cxcywh(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]


def to_xyxy(c):
    return np.c_[c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2]


def window_mean(frames, vals, back, fwd):
    """Среднее по наблюдениям трека в окне [t − back, t + fwd] (по номерам кадров)."""
    out = np.empty_like(vals)
    lo = np.searchsorted(frames, frames - back, "left"); hi = np.searchsorted(frames, frames + fwd, "right")
    cs = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, 0)])
    for i in range(len(frames)):
        out[i] = (cs[hi[i]] - cs[lo[i]]) / (hi[i] - lo[i])
    return out


def transform(outputs, variant, det):
    if variant == "base" or len(outputs) == 0:
        return outputs
    o = outputs.copy()
    if variant == "det":
        from scripts.oracle_detection_ladder import iou_mat
        for t in np.unique(o[:, 0]).astype(int):
            rows = np.flatnonzero(o[:, 0] == t); dm = det["frame_idx"] == t
            db, dc = det["xyxy"][dm], det["cls"][dm].astype(int)
            if not len(db):
                continue
            m = np.where(o[rows, 7][:, None].astype(int) == dc[None, :], iou_mat(o[rows, 1:5], db), 0.0)
            j = m.argmax(1); ok = m[np.arange(len(rows)), j] >= 0.6
            o[rows[ok], 1:5] = db[j[ok]]
        return o
    res = []
    for tid in np.unique(o[:, 5]):
        r = o[o[:, 5] == tid]; r = r[np.argsort(r[:, 0])]
        f = r[:, 0].astype(int); c = to_cxcywh(r[:, 1:5])
        if variant.startswith("ma"):
            w = int(variant.split("_")[0][2:]); c = window_mean(f, c, w // 2, w // 2)
        elif variant.startswith("lag"):
            L = int(variant[3:]); c = window_mean(f, c, L, L)
        r[:, 1:5] = to_xyxy(c)
        if variant.endswith("interp"):
            extra = []
            for a, b in zip(range(len(f) - 1), range(1, len(f))):
                gap = f[b] - f[a]
                if 1 < gap <= MAX_GAP:
                    for k in range(1, gap):
                        w_ = k / gap; row = r[a].copy(); row[0] = f[a] + k
                        row[1:5] = (1 - w_) * r[a, 1:5] + w_ * r[b, 1:5]; extra.append(row)
            if extra:
                r = np.vstack([r, np.array(extra, r.dtype)])
        res.append(r)
    return np.vstack(res)


def run(task):
    tracker, variant, vid, root, split = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import load_detection_cache, save_json, scalabel_frame

    per_path = OUT / tracker / variant / "per_video" / f"{vid}.json"
    if per_path.exists():
        return tracker, variant, vid, json.loads(per_path.read_text(encoding="utf-8"))
    outputs = np.load(SRC / tracker / f"{vid}.npz")["outputs"]
    det = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz") if variant == "det" else None
    o = transform(outputs, variant, det)
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    by = {}
    for row in o:
        by.setdefault(int(row[0]), []).append(row[1:8])
    rows = [scalabel_frame(vid, truth[i], np.asarray(by.get(int(m["video_frame_index"]), []), np.float64).reshape(-1, 7)) for i, m in enumerate(matches)]
    res_dir = OUT / tracker / variant / split; res_dir.mkdir(parents=True, exist_ok=True)
    save_json(res_dir / f"{vid}.json", rows)
    met = evaluate_hota(res_dir, [vid], split=split)
    r = {k: met.get(k) for k in METRICS}
    per_path.parent.mkdir(parents=True, exist_ok=True); per_path.write_text(json.dumps(r), encoding="utf-8")
    return tracker, variant, vid, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from evaluation.hota_eval import evaluate_hota

    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    sets = {"test": (sel["test"], PROJECT_ROOT / "runs/train_cache", "train"), "val7": (VAL7, PROJECT_ROOT / "runs/video_rate", "val")}
    if args.limit:
        sets = {k: (v[0][:args.limit], v[1], v[2]) for k, v in sets.items()}
    cond = {}
    for s, (vids, _, split) in sets.items():
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        for c in ("normal", "night"):
            for v in filter_videos_by_subset(attrs, vids, c):
                cond.setdefault(v, c)
    tasks = [(tr, var, v, str(root), split) for tr in TRACKERS for var in VARIANTS for s, (vids, root, split) in sets.items() for v in vids]
    per = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (tr, var, vid, r) in enumerate(pool.map(run, tasks), 1):
            per.setdefault(tr, {}).setdefault(var, {})[vid] = r
            if k % 40 == 0:
                print(f"[RUN] {k}/{len(tasks)}", flush=True)
    report = {"sets": {}}
    rng = np.random.default_rng(0)
    for tr in TRACKERS:
        for s, (vids, root, split) in sets.items():
            groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
            summ, boot = {}, {}
            for var in VARIANTS:
                summ[var] = {}
                for g, vs in groups.items():
                    if vs:
                        agg = evaluate_hota(OUT / tr / var / split, vs, split=split); summ[var][g] = {k: agg.get(k) for k in METRICS}
                if var != "base":
                    for met in ("HOTA", "DetA", "AssA", "IDF1"):
                        d = np.array([per[tr][var][v][met] - per[tr]["base"][v][met] for v in vids], np.float64)
                        b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                        boot[f"{var}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
            report["sets"][f"{tr}|{s}"] = {"summary": summ, "bootstrap_vs_base": boot}
            print(f"== {tr} | {s}")
            for var in VARIANTS:
                a = summ[var]["all"]; bb = boot.get(f"{var}|HOTA")
                print(f"  {var:11s} HOTA {a['HOTA']:.3f} DetA {a['DetA']:.3f} AssA {a['AssA']:.3f} IDF1 {a['IDF1']:.3f}" + (f"  Δ {bb['mean']:+.3f} [{bb['ci95'][0]:+.3f}; {bb['ci95'][1]:+.3f}] {bb['better']}/{bb['worse']}" if bb else ""))
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
