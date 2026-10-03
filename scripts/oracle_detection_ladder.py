"""Д1: лестница оракулов детекции — сколько HOTA даёт каждая сторона обнаружения (карточка detection_oracle_ladder_v1).

Оракулы правят детекции ТОЛЬКО на размеченных кадрах (5 Гц, по эталону box_track_20); трекер — стенд
(XYWHByteTrack, bench_freeze_v1, без каскада и сшивки) на всех сырых кадрах 30 Гц; HOTA — TrackEval.
  base       — без изменений (воспроизведение стенда);
  thr        — объект ступени ПОРОГ (есть кандидат того же класса IoU ≥ 0,5, но лучшая уверенность < 0,25): его лучшему кандидату ставится 0,9;
  thr_ctl    — контроль: столько же случайных НЕсопоставленных кандидатов < 0,25 того же кадра поднимаются до 0,9;
  thr_young  — как thr, но только для возраста объекта 0–1 (рождение);
  fp         — удалены кандидаты ≥ 0,1, не совпадающие ни с одним эталоном того же класса (IoU < 0,5);
  box        — рамки сопоставленных кандидатов (жадно по уверенности) заменены эталонными;
  miss       — для эталонов без кандидата того же класса IoU ≥ 0,5 добавлена эталонная рамка с уверенностью 0,9;
  all        — thr + fp + box + miss.
Отчёт: test (35, train_cache) и val7; ночь/день по атрибутам; бутстреп по видео разности с base.

Запуск: python scripts/oracle_detection_ladder.py --workers 10
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

OUT = PROJECT_ROOT / "runs" / "oracle_ladder"
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
CATS = {"pedestrian": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7}
VARIANTS = ["base", "thr", "thr_ctl", "thr_young", "fp", "box", "miss", "all"]
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    sa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); sb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1e-9)


def modify(det, truth, matches, variant, rng):
    """Вернуть изменённые xyxy, conf, cls, frame_idx."""
    xyxy, conf, cls, fidx = det["xyxy"].copy(), det["conf"].copy(), det["cls"].copy(), det["frame_idx"].copy()
    if variant == "base":
        return xyxy, conf, cls, fidx
    keep = np.ones(len(conf), bool)
    add_b, add_c, add_k, add_f = [], [], [], []
    seen = {}
    for m, fr in zip(matches, truth):
        r = int(m["video_frame_index"])
        gb, gc, ages = [], [], []
        for l in fr.get("labels") or []:
            bb, cat = l.get("box2d"), l.get("category")
            if not bb or cat not in CATS:
                continue
            gb.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); gc.append(CATS[cat])
            ages.append(seen.get(l["id"], 0)); seen[l["id"]] = seen.get(l["id"], 0) + 1
        if not gb:
            continue
        gb, gc, ages = np.array(gb, np.float32), np.array(gc), np.array(ages)
        sel = np.flatnonzero(fidx == r)
        m_iou = np.where(gc[:, None] == cls[sel][None, :].astype(int), iou_mat(gb, xyxy[sel]), 0.0) if len(sel) else np.zeros((len(gb), 0))
        hit = (m_iou >= 0.5) if m_iou.shape[1] else np.zeros((len(gb), 0), bool)
        if variant in ("thr", "thr_young", "all", "thr_ctl"):
            n_lift = 0
            for g in range(len(gb)):
                if not hit.shape[1] or not hit[g].any():
                    continue
                cand = np.flatnonzero(hit[g]); best = cand[np.argmax(conf[sel][cand])]
                if conf[sel][best] < 0.25 and (variant != "thr_young" or ages[g] <= 1):
                    n_lift += 1
                    if variant != "thr_ctl":
                        conf[sel[best]] = 0.9
            if variant == "thr_ctl" and n_lift:
                unmatched = [j for j in range(len(sel)) if (not hit.shape[0] or not hit[:, j].any()) and conf[sel[j]] < 0.25]
                for j in rng.choice(unmatched, min(n_lift, len(unmatched)), replace=False) if unmatched else []:
                    conf[sel[j]] = 0.9
        if variant in ("fp", "all") and len(sel):
            unmatched = ~hit.any(0) if hit.shape[0] else np.ones(len(sel), bool)
            keep[sel[unmatched & (conf[sel] >= 0.1)]] = False
        if variant in ("box", "all") and len(sel):
            used = np.zeros(len(gb), bool)
            for j in np.argsort(-conf[sel]):
                col = np.where(used, 0.0, m_iou[:, j]); g = int(col.argmax())
                if col[g] >= 0.5:
                    used[g] = True; xyxy[sel[j]] = gb[g]
        if variant in ("miss", "all"):
            for g in range(len(gb)):
                if not hit.shape[1] or not hit[g].any():
                    add_b.append(gb[g]); add_c.append(0.9); add_k.append(gc[g]); add_f.append(r)
    xyxy, conf, cls, fidx = xyxy[keep], conf[keep], cls[keep], fidx[keep]
    if add_b:
        xyxy = np.concatenate([xyxy, np.array(add_b, np.float32)]); conf = np.concatenate([conf, np.array(add_c, conf.dtype)])
        cls = np.concatenate([cls, np.array(add_k, cls.dtype)]); fidx = np.concatenate([fidx, np.array(add_f, fidx.dtype)])
    return xyxy, conf, cls, fidx


def run(task):
    variant, vid, root, split = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ultralytics.trackers.basetrack import BaseTrack
    from evaluation.hota_eval import evaluate_hota
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import load_detection_cache, save_json, scalabel_frame
    from tracking.motion_source_bytetrack import MotionSourceByteTrack
    from tracking.offline_association import _DetSlice, make_tracker_args

    res_dir = OUT / variant / split
    res_path = res_dir / f"{vid}.json"
    per_path = OUT / variant / "per_video" / f"{vid}.json"
    if per_path.exists():
        return variant, vid, json.loads(per_path.read_text(encoding="utf-8"))
    det = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz")
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(abs(hash(vid)) % (2 ** 32))
    xyxy, conf, cls, fidx = modify(det, truth, matches, variant, rng)
    order = np.argsort(fidx, kind="stable"); xyxy, conf, cls, fidx = xyxy[order], conf[order], cls[order], fidx[order]
    bounds = np.searchsorted(fidx, np.arange(int(det["frame_count"]) + 1))
    BaseTrack.reset_id()
    tracker = MotionSourceByteTrack(make_tracker_args(**XYWH_CONFIG))
    by_frame = {}
    for t in range(int(det["frame_count"])):
        a, b = bounds[t], bounds[t + 1]
        out = np.asarray(tracker.update(_DetSlice(xyxy[a:b], conf[a:b], cls[a:b])))
        if len(out):
            by_frame[t] = out[:, :7].astype(np.float64)
    rows = [scalabel_frame(vid, truth[i], by_frame.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)]
    res_dir.mkdir(parents=True, exist_ok=True); save_json(res_path, rows)
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
            if k % 20 == 0:
                print(f"[RUN] {k}/{len(tasks)}", flush=True)
    report = {"sets": {}, "cond": cond}
    rng = np.random.default_rng(0)
    for s, (vids, root, split) in sets.items():
        rs = {}
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        for var in VARIANTS:
            rs[var] = {}
            for g, vs in groups.items():
                if not vs:
                    continue
                agg = evaluate_hota(OUT / var / split, vs, split=split)
                rs[var][g] = {k: agg.get(k) for k in METRICS}
        boot = {}
        for var in VARIANTS[1:]:
            for g, vs in groups.items():
                if not vs:
                    continue
                for met in ("HOTA", "DetA"):
                    d = np.array([per[var][v][met] - per["base"][v][met] for v in vs], np.float64)
                    b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    boot[f"{var}|{g}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]}
        report["sets"][s] = {"summary": rs, "bootstrap_vs_base": boot, "videos": {g: len(vs) for g, vs in groups.items()}}
        print(f"== {s}: видео {report['sets'][s]['videos']}")
        for var in VARIANTS:
            line = " | ".join(f"{g}: HOTA {rs[var][g]['HOTA']:.3f} DetA {rs[var][g]['DetA']:.3f}" for g in rs[var])
            print(f"  {var:10s} {line}")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
