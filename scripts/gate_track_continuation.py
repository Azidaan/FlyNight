"""Гейт продолжения треков: возвращает ли политика продолжения потери на пороге (карточка track_continuation_v1).

Основание — past_view_oracle_v1: у ~1/3 реальных потерь на пороге объект уверенно виден в прошлых 5 кадрах,
но трекер-стенд его не выдаёт (трек ушёл в «потерянные», ByteTrack выдаёт только сопоставленные в кадре).
Варианты стенда (XYWHByteTrack, bench_freeze_v1; меняется одно условие, кроме *_combo):
  base      — стенд (h 0,25 / l 0,1 / буфер 30 / m 0,8);
  low05     — порог второй волны 0,05;
  low01     — порог второй волны 0,01 (все кандидаты кэша);
  coast3    — дополнительно выдаются потерянные треки ≤ 3 кадров после последнего сопоставления (рамка — предсказание Калмана);
  coast6    — то же, ≤ 6 кадров;
  combo     — low05 + coast3.
Test 35 и val7; HOTA TrackEval; бутстреп по видео. Разметка трекеру не подаётся.

Запуск: python scripts/gate_track_continuation.py --workers 10
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

from scripts.oracle_detection_ladder import VAL7  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "gate_continuation"
VARIANTS = {"base": ({}, 0), "low05": ({"track_low_thresh": 0.05}, 0), "low01": ({"track_low_thresh": 0.01}, 0),
            "coast3": ({}, 3), "coast6": ({}, 6), "combo": ({"track_low_thresh": 0.05}, 3)}
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def run(task):
    variant, vid, root, split = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ultralytics.trackers.basetrack import BaseTrack
    from evaluation.hota_eval import evaluate_hota
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, save_json, scalabel_frame
    from tracking.motion_source_bytetrack import MotionSourceByteTrack
    from tracking.offline_association import _DetSlice, make_tracker_args

    per_path = OUT / variant / "per_video" / f"{vid}.json"
    if per_path.exists():
        return variant, vid, json.loads(per_path.read_text(encoding="utf-8"))
    over, coast = VARIANTS[variant]
    det = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz")
    BaseTrack.reset_id()
    tracker = MotionSourceByteTrack(make_tracker_args(**{**XYWH_CONFIG, **over}))
    by = {}
    for t in range(int(det["frame_count"])):
        xyxy, conf, cls = detections_for_frame(det, t)
        out = np.asarray(tracker.update(_DetSlice(xyxy, conf, cls)))
        rows = [out[:, :7]] if len(out) else []
        if coast:
            extra = [[*tr.xyxy.tolist(), tr.track_id, tr.score, tr.cls] for tr in tracker.lost_stracks
                     if tr.is_activated and 0 < tracker.frame_id - tr.end_frame <= coast]
            if extra:
                rows.append(np.asarray(extra, np.float64))
        if rows:
            by[t] = np.vstack(rows).astype(np.float64)
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    rows = [scalabel_frame(vid, truth[i], by.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)]
    res_dir = OUT / variant / split; res_dir.mkdir(parents=True, exist_ok=True)
    save_json(res_dir / f"{vid}.json", rows)
    met = evaluate_hota(res_dir, [vid], split=split)
    r = {k: met.get(k) for k in METRICS}
    per_path.parent.mkdir(parents=True, exist_ok=True); per_path.write_text(json.dumps(r), encoding="utf-8")
    return variant, vid, r


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
    tasks = [(var, v, str(root), split) for var in VARIANTS for s, (vids, root, split) in sets.items() for v in vids]
    per = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (var, vid, r) in enumerate(pool.map(run, tasks), 1):
            per.setdefault(var, {})[vid] = r
            if k % 40 == 0:
                print(f"[RUN] {k}/{len(tasks)}", flush=True)
    rng = np.random.default_rng(0)
    report = {"sets": {}}
    for s, (vids, root, split) in sets.items():
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        summ, boot = {}, {}
        for var in VARIANTS:
            summ[var] = {g: {k: evaluate_hota(OUT / var / split, vs, split=split).get(k) for k in METRICS} for g, vs in groups.items() if vs}
            if var == "base":
                continue
            for g, vs in groups.items():
                for met in ("HOTA", "DetA", "AssA", "IDF1", "IDSw"):
                    d = np.array([per[var][v][met] - per["base"][v][met] for v in vs], np.float64)
                    if not len(d):
                        continue
                    b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    boot[f"{var}|{g}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "better": int(((d < 0) if met == "IDSw" else (d > 0)).sum()), "worse": int(((d > 0) if met == "IDSw" else (d < 0)).sum())}
        report["sets"][s] = {"summary": summ, "bootstrap_vs_base": boot}
        print(f"== {s}")
        for var in VARIANTS:
            a = summ[var]["all"]; bb = boot.get(f"{var}|all|HOTA"); bn = boot.get(f"{var}|night|HOTA")
            print(f"  {var:7s} HOTA {a['HOTA']:.3f} DetA {a['DetA']:.3f} AssA {a['AssA']:.3f} IDF1 {a['IDF1']:.3f} IDSw {a['IDSw']:.0f}"
                  + (f"  Δ {bb['mean']:+.3f} [{bb['ci95'][0]:+.3f}; {bb['ci95'][1]:+.3f}] {bb['better']}/{bb['worse']}" if bb else "")
                  + (f"  ночь Δ {bn['mean']:+.3f}" if bn else ""))
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
