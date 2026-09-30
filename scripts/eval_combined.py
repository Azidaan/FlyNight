"""Итоговый трекер линии B: каскад буферов IoU + отложенное рождение + сшивка по геометрии (карточка line_b_combined_v1).

Трекеры: стенд; каскад 0,3→0,6; отложенное рождение N = 2 (голова q_birth runs/birth_train); каскад + отложенное.
Каждый — без сшивки и со сшивкой обрывков по геометрии (голова geometry из runs/fly_reid_ablation/models.pkl,
обученная на парах стенда train-роликов; жизни — непрерывные отрезки выхода трекера). Порог сшивки —
максимум IDF1 на dev для каждого трекера; отчёт — test (35) и val7; бутстреп по видео.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
TRACKERS = {"stand": {}, "cascade": {"buffer": 0.3, "cascade_buffer": 0.6}, "deferred": {"deferred": True}, "cascade_deferred": {"buffer": 0.3, "cascade_buffer": 0.6, "deferred": True}}
THRESHOLDS = (0.05, 0.1, 0.2, 0.3)
MAX_GAP, MAX_DIST = 60, 3.0
MIN_LIFE = 2
METRICS = ("HOTA", "IDF1", "AssA", "DetA", "IDSw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "line_b_combined")
    parser.add_argument("--stitch-model", type=Path, default=PROJECT_ROOT / "runs" / "fly_reid_ablation" / "models.pkl")
    parser.add_argument("--birth-head", type=Path, default=PROJECT_ROOT / "runs" / "birth_train")
    parser.add_argument("--confirm-hits", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def run_tracker(task: tuple) -> dict:
    """Прогнать трекер по ролику, сохранить все выходы по кадрам (frame, x1, y1, x2, y2, tid, score, cls)."""
    name, spec, video_id, root, videos, out_dir, birth_dir, confirm_hits = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    from ultralytics.trackers.basetrack import BaseTrack
    from scripts.build_event_dataset import CLASS_IDS, MIN_CONF
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, load_motion_cache
    from tracking.birth_policy import BirthHead
    from tracking.decision_log import IndexedDetections
    from tracking.motion_source_bytetrack import MotionSourceByteTrack, make_deferred_cascade
    from tracking.offline_association import _DetSlice, make_tracker_args

    path = Path(out_dir) / name / f"{video_id}.npz"
    if path.exists():
        return {"name": name, "video_id": video_id}
    detections = load_detection_cache(Path(root) / video_id / "detections" / "raw_video.npz")
    BaseTrack.reset_id()
    deferred = spec.get("deferred", False)
    if deferred:
        head = BirthHead.from_json(Path(birth_dir) / "head.json")
        tau = json.loads((Path(birth_dir) / "head_info.json").read_text(encoding="utf-8"))["tau_f1"]
        tracker = make_deferred_cascade(make_tracker_args(**XYWH_CONFIG), buffer=spec.get("buffer", 0.0), cascade_buffer=spec.get("cascade_buffer", 0.0), head=head, tau=tau, confirm_hits=confirm_hits, class_ids=CLASS_IDS)
        points, vectors = load_motion_cache(Path(root) / video_id / "motion" / "dis_quarter.npz", 1.0)
        capture = cv2.VideoCapture(str(Path(videos) / f"{video_id}.mov"))
        frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    else:
        tracker = MotionSourceByteTrack(make_tracker_args(**XYWH_CONFIG), buffer=spec.get("buffer", 0.0), cascade_buffer=spec.get("cascade_buffer", 0.0))
    rows, prev_boxes = [], np.zeros((0, 4), np.float32)
    for t in range(int(detections["frame_count"])):
        xyxy, conf, cls = detections_for_frame(detections, t)
        if deferred:
            ok, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else np.zeros((frame_size[1], frame_size[0]), np.uint8)
            prev_prop = np.asarray([propagate_box(b, points, vectors, t - 1, t)[0] for b in prev_boxes], np.float32).reshape(-1, 4) if t > 0 and len(prev_boxes) else np.zeros((0, 4), np.float32)
            tracker.set_frame_context(gray=gray, prev_propagated=prev_prop, frame_size=frame_size)
            output = tracker.update_with_log(IndexedDetections(xyxy, conf, cls))["output"]
            prev_boxes = xyxy[conf > MIN_CONF]
        else:
            output = tracker.update(_DetSlice(xyxy, conf, cls))
        output = np.asarray(output)
        if len(output):
            rows.append(np.column_stack([np.full(len(output), t), output[:, :7]]).astype(np.float32))
    if deferred:
        capture.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, outputs=np.concatenate(rows) if rows else np.zeros((0, 8), np.float32), frame_count=np.asarray(int(detections["frame_count"])))
    return {"name": name, "video_id": video_id}


def lives_from_outputs(outputs: np.ndarray) -> list[dict]:
    lives = []
    for tid in np.unique(outputs[:, 5]):
        rows = outputs[outputs[:, 5] == tid]; rows = rows[np.argsort(rows[:, 0])]
        cut = np.flatnonzero(np.diff(rows[:, 0]) > 1) + 1
        for seg in np.split(rows, cut):
            if len(seg) < MIN_LIFE:
                continue  # как жизни глаза в build_fly_signatures: трек, не доживший до следующего кадра, в сшивку не входит
            lives.append({"tid": int(tid), "start": int(seg[0, 0]), "end": int(seg[-1, 0]), "first_box": seg[0, 1:5].astype(np.float64), "last_box": seg[-1, 1:5].astype(np.float64), "cls": int(seg[0, 7]), "n": min(len(seg), 5)})
    return lives


def stitch_eval(task: tuple) -> dict:
    name, video_id, root, label_split, out_dir, thr, model_path, run_dir = task
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import scalabel_frame

    outputs = np.load(Path(run_dir) / name / f"{video_id}.npz")["outputs"]
    parent = {}
    if thr < 1.0 and len(outputs):
        with open(model_path, "rb") as fh:
            model, idx = pickle.load(fh)["geometry"]
        lives = lives_from_outputs(outputs)
        feats, pairs = [], []
        for a, A in enumerate(lives):
            for b, B in enumerate(lives):
                if A["tid"] == B["tid"]:
                    continue
                gap = B["start"] - A["end"]
                if not (0 < gap <= MAX_GAP):
                    continue
                sa = np.sqrt(max((A["last_box"][2] - A["last_box"][0]) * (A["last_box"][3] - A["last_box"][1]), 1.0))
                sb = np.sqrt(max((B["first_box"][2] - B["first_box"][0]) * (B["first_box"][3] - B["first_box"][1]), 1.0))
                dist = np.linalg.norm((A["last_box"][:2] + A["last_box"][2:]) / 2 - (B["first_box"][:2] + B["first_box"][2:]) / 2) / sa
                if dist > MAX_DIST:
                    continue
                feats.append([gap / 30.0, dist, float(np.log(sb / sa)), float(A["cls"] == B["cls"]), A["n"] / 5, B["n"] / 5]); pairs.append((a, b))
        if feats:
            probs = model.predict_proba(np.asarray(feats, np.float64))[:, 1]
            frames_of = {}
            for r in outputs:
                frames_of.setdefault(int(r[5]), set()).add(int(r[0]))
            merged = {t: set(f) for t, f in frames_of.items()}
            def root_of(t):
                seen = set()
                while t in parent and t not in seen:
                    seen.add(t); t = parent[t]
                return t
            used_a, used_b = set(), set()
            for k in np.argsort(-probs):
                if probs[k] <= thr:
                    break
                a, b = pairs[k]
                ra, rb = root_of(lives[a]["tid"]), root_of(lives[b]["tid"])
                if a in used_a or b in used_b or ra == rb or merged.get(ra, set()) & merged.get(rb, set()):
                    continue
                used_a.add(a); used_b.add(b); parent[rb] = ra; merged[ra] = merged.get(ra, set()) | merged.pop(rb, set())
    def root_final(t):
        seen = set()
        while t in parent and t not in seen:
            seen.add(t); t = parent[t]
        return t
    matches = json.loads((Path(root) / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / label_split / f"{video_id}.json").read_text(encoding="utf-8"))
    by_frame = {}
    for r in outputs:
        by_frame.setdefault(int(r[0]), []).append(r[1:8].copy())
    rows = []
    for i, m in enumerate(matches):
        out = np.asarray(by_frame.get(int(m["video_frame_index"]), []), np.float64).reshape(-1, 7)
        if len(out):
            out[:, 4] = [root_final(int(t)) for t in out[:, 4]]
        rows.append(scalabel_frame(video_id, truth[i], out))
    res_dir = Path(out_dir); res_dir.mkdir(parents=True, exist_ok=True)
    save_json(res_dir / f"{video_id}.json", rows)
    metrics = evaluate_hota(res_dir, [video_id], split=label_split)
    return {"video_id": video_id, "links": len(parent), "metrics": {k: metrics.get(k) for k in METRICS}}


def main() -> int:
    args = parse_args()
    from evaluation.hota_eval import evaluate_hota

    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    splits = {"dev": (selection["dev"], args.train_cache, PROJECT_ROOT / "data/bdd100k/videos/train", "train"),
              "test": (selection["test"], args.train_cache, PROJECT_ROOT / "data/bdd100k/videos/train", "train"),
              "val7": (VAL7, PROJECT_ROOT / "runs/video_rate", PROJECT_ROOT / "data/bdd100k/videos/val", "val")}
    runs = args.output / "outputs"
    tasks = [(name, spec, v, str(root), str(videos), str(runs), str(args.birth_head), args.confirm_hits) for name, spec in TRACKERS.items() for s, (vids, root, videos, _) in splits.items() for v in vids]
    print(f"[PLAN] трекеров {len(TRACKERS)} × роликов {sum(len(s[0]) for s in splits.values())} = {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(run_tracker, tasks, chunksize=1), 1):
            if k % 40 == 0:
                print(f"[TRACK] {k}/{len(tasks)}", flush=True)

    def evaluate(name: str, split: str, thr: float) -> dict:
        vids, root, _, lsplit = splits[split]
        out_dir = args.output / "stitch" / f"{name}_t{thr:.2f}" / split
        ts = [(name, v, str(root), lsplit, str(out_dir), thr, str(args.stitch_model), str(runs)) for v in vids]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            per = {r["video_id"]: r for r in pool.map(stitch_eval, ts)}
        agg = evaluate_hota(out_dir, list(vids), split=lsplit)
        return {"summary": {k: agg.get(k) for k in METRICS}, "links": int(sum(r["links"] for r in per.values())), "per_video": {v: r["metrics"] for v, r in per.items()}}

    report = {"trackers": TRACKERS, "results": {}}
    for name in TRACKERS:
        base = {s: evaluate(name, s, 1.0) for s in splits}
        dev_grid = {thr: evaluate(name, "dev", thr) for thr in THRESHOLDS}
        best = max(dev_grid, key=lambda t: dev_grid[t]["summary"]["IDF1"])
        stitched = {"dev": dev_grid[best], "test": evaluate(name, "test", best), "val7": evaluate(name, "val7", best)}
        report["results"][name] = {"no_stitch": base, "stitch": stitched, "threshold": best, "dev_grid": {str(t): r["summary"] for t, r in dev_grid.items()}}
        for tag, r in (("", base), ("+stitch", stitched)):
            print(f"[SUM] {name + tag:26s} " + " | ".join(f"{s}: HOTA {r[s]['summary']['HOTA']:.3f} IDF1 {r[s]['summary']['IDF1']:.3f} IDSw {r[s]['summary']['IDSw']:.0f}" for s in splits) + (f" (порог {best}, связей test {r['test']['links']})" if tag else ""), flush=True)
    ref = report["results"]["stand"]["no_stitch"]
    boot = {}
    for name in TRACKERS:
        for tag in ("no_stitch", "stitch"):
            cand = report["results"][name][tag]
            for s in ("test", "val7"):
                for met in ("HOTA", "IDF1", "IDSw"):
                    d = np.asarray([cand[s]["per_video"][v][met] - ref[s]["per_video"][v][met] for v in ref[s]["per_video"]], np.float64)
                    rng = np.random.default_rng(42); b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                    boot[f"{name}|{tag}|{s}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "better": int(((d < 0) if met == "IDSw" else (d > 0)).sum()), "worse": int(((d > 0) if met == "IDSw" else (d < 0)).sum())}
    report["bootstrap_vs_stand"] = boot
    for k, v in boot.items():
        if k.endswith("|HOTA") and "|test|" in k:
            print(f"[BOOT] {k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}; {v['ci95'][1]:+.3f}] {v['better']}/{v['worse']}", flush=True)
    save_json(args.output / "report.json", report)
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
