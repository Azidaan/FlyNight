"""Построить четыре выборки событий трекера (карточка event_dataset_v1, цель 4).

XYWHByteTrack замороженного стенда с журналом решений идёт по всем кадрам;
разметка читается только после решений и только на опорных кадрах.
Признаки не содержат GT. Правила меток — tracking/event_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import (  # noqa: E402
    box_pair_geometry,
    detections_with_truth_status,
    propagate_box,
    truth_arrays,
)
from scripts.freeze_bench import XYWH_CONFIG  # noqa: E402
from scripts.run_bdd_video_rate_experiment import CLASS_NAMES, detections_for_frame, load_detection_cache, load_motion_cache, save_json  # noqa: E402
from tracking.decision_log import AuditedXYWHByteTrack, IndexedDetections  # noqa: E402
from tracking.event_dataset import IdentityTimeline, admissible_candidates, alive_label, birth_label, edge_distance, frames_to_exit, image_box_stats  # noqa: E402
from tracking.flyvis_tracker import box_iou  # noqa: E402
from tracking.offline_association import make_tracker_args  # noqa: E402

VIDEO_IDS = (
    "b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117",
    "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd",
)
CLASS_IDS = sorted(CLASS_NAMES)
MIN_CONF = 0.1          # ниже трекер не видит детекции
DETECTION_IOU = 0.5     # сопоставление с GT
BIRTH_HORIZON = 3       # опорных кадров
ALIVE_WINDOW = 6        # опорных кадров
BIRTH_MAX_GAP = 5       # исходных кадров до первого опорного
CANDIDATE_DISTANCE = 2.5

GEOMETRY = ("width", "height", "area_fraction", "center_x", "center_y", "edge_distance")
IMAGE = ("img_mean", "img_std", "img_dark", "img_bright")
CLASS_ONE_HOT = tuple(f"class_{c}" for c in CLASS_IDS)
DET_FEATURES = ("conf",) + CLASS_ONE_HOT + GEOMETRY + IMAGE + ("ctx_max_iou_tracked", "ctx_tracks_near", "dis_prev_max_iou", "dis_prev_exists")
MATCH_FEATURES = ("pair_iou", "pair_center_distance", "pair_size_log_change", "det_conf") + CLASS_ONE_HOT + ("track_age", "track_hits", "track_since_update", "track_lost", "track_speed", "track_last_score", "n_candidates", "iou_margin", "n_competing_tracks", "dis_track_iou")
BIRTH_FEATURES = ("conf",) + CLASS_ONE_HOT + GEOMETRY + IMAGE + ("ctx_max_iou_tracked", "ctx_tracks_near", "dis_prev_max_iou", "dis_prev_exists")
ALIVE_FEATURES = ("track_age", "track_hits", "track_last_score", "track_speed", "velocity_x", "velocity_y") + CLASS_ONE_HOT + GEOMETRY + ("frames_to_exit", "dis_magnitude") + IMAGE
DIS_NAMES = {"q_det": ("dis_prev_max_iou", "dis_prev_exists"), "q_match": ("dis_track_iou",), "q_birth": ("dis_prev_max_iou", "dis_prev_exists"), "q_alive": ("dis_magnitude",)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "event_dataset")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _xywh_to_xyxy(xywh) -> np.ndarray:
    """Предсказание фильтра может выродиться (w или h ≤ 0); рамка санируется до 1 px."""
    cx, cy, w, h = (float(v) for v in xywh)
    if not np.isfinite([cx, cy, w, h]).all():
        cx, cy, w, h = 0.0, 0.0, 1.0, 1.0
    w, h = max(w, 1.0), max(h, 1.0)
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def _geometry(box, frame_size) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = frame_size
    return [(x2 - x1) / width, (y2 - y1) / height, (x2 - x1) * (y2 - y1) / (width * height), (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, edge_distance(box, frame_size)]


def _image(gray, box) -> list[float]:
    s = image_box_stats(gray, box)
    return [s["mean"] / 255.0, s["std"] / 255.0, s["dark_fraction"], s["bright_fraction"]]


def _one_hot(class_id: int) -> list[float]:
    return [float(class_id == c) for c in CLASS_IDS]


def _crowd_boxes(frame: dict) -> np.ndarray:
    boxes = [[l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]] for l in frame.get("labels", []) if l.get("attributes", {}).get("crowd", False) and l.get("box2d")]
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def _context(box, tracked_boxes: np.ndarray) -> tuple[float, float]:
    if not len(tracked_boxes):
        return 0.0, 0.0
    overlaps = box_iou(np.asarray(box, np.float32)[None, :], tracked_boxes)[0]
    return float(overlaps.max()), float((overlaps > 0.3).sum())


def _dis_prev(box, prev_boxes_propagated: np.ndarray) -> tuple[float, float]:
    if not len(prev_boxes_propagated):
        return 0.0, 0.0
    overlaps = box_iou(np.asarray(box, np.float32)[None, :], prev_boxes_propagated)[0]
    return float(overlaps.max()), float(overlaps.max() >= 0.5)


def _dis_magnitude(box, points, vectors_t) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    inside = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    return float(np.linalg.norm(vectors_t[inside], axis=1).mean()) if inside.any() else 0.0


def build_video(video_id: str, args: argparse.Namespace) -> dict:
    output = args.output / video_id
    if (output / "manifest.json").exists() and not args.force:
        print(f"[CACHE] {video_id}", flush=True)
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    detections = load_detection_cache(args.root / video_id / "detections" / "raw_video.npz")
    matches = json.loads((args.root / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
    ref_of_raw = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
    truth = json.loads((args.labels / f"{video_id}.json").read_text(encoding="utf-8"))
    n_ref = len(truth)
    dis_points, dis_vectors = load_motion_cache(args.root / video_id / "motion" / "dis_quarter.npz", 1.0)
    capture = cv2.VideoCapture(str(args.videos / f"{video_id}.mov"))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_id}")
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    BaseTrack.reset_id()
    tracker = AuditedXYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    timeline = IdentityTimeline()
    presence: dict[str, set[int]] = defaultdict(set)
    for r, frame in enumerate(truth):
        for gt_id in truth_arrays(frame)[2]:
            presence[gt_id].add(r)

    rows = {"q_det": [], "q_match": [], "q_birth": [], "q_alive": []}
    meta = {"q_det": [], "q_match": [], "q_birth": [], "q_alive": []}
    pending_births: list[dict] = []
    pending_alive: list[dict] = []
    counters = Counter()
    prev_boxes = np.zeros((0, 4), np.float32)
    prev_predictions: dict[int, np.ndarray] = {}
    last_ref = -1
    started = time.perf_counter()
    t = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xyxy, conf, cls = detections_for_frame(detections, t)
        log = tracker.update_with_log(IndexedDetections(xyxy, conf, cls))
        visible = conf > MIN_CONF
        predictions = log["predictions"]
        pred_boxes = {p["track_id"]: _xywh_to_xyxy(p["xywh"]) for p in predictions}
        tracked_boxes = np.asarray([pred_boxes[p["track_id"]] for p in predictions if p["state"] == "tracked"], np.float32).reshape(-1, 4)
        prev_propagated = np.asarray([propagate_box(b, dis_points, dis_vectors, t - 1, t)[0] for b in prev_boxes], np.float32).reshape(-1, 4) if t > 0 and len(prev_boxes) else np.zeros((0, 4), np.float32)
        output_boxes = {int(row[4]): row[:4].astype(np.float32) for row in log["output"]}

        # --- рождения: снимок признаков в момент события ------------------------------
        for e in log["events"]:
            if e["event"] == "created":
                j = e["det_index"]
                ctx = _context(xyxy[j], tracked_boxes)
                dis = _dis_prev(xyxy[j], prev_propagated)
                pending_births.append({"track_id": e["track_id"], "frame": t, "det_index": j, "class_id": int(cls[j]), "det_box": xyxy[j].copy(),
                                       "features": [float(conf[j])] + _one_hot(int(cls[j])) + _geometry(xyxy[j], frame_size) + _image(gray, xyxy[j]) + [ctx[0], ctx[1], dis[0], dis[1]]})
            elif e["event"] == "state_transition" and e["to"] == "lost":
                p = next((q for q in predictions if q["track_id"] == e["track_id"]), None)
                if p is None:
                    counters["alive_no_prediction"] += 1
                    continue
                gt_id = timeline.upto(e["track_id"], last_ref) if last_ref >= 0 else None
                if gt_id is None:
                    counters["alive_identity_unknown"] += 1
                    continue
                box = pred_boxes[e["track_id"]]
                v = p["velocity"]
                pending_alive.append({"track_id": e["track_id"], "frame": t, "last_ref": last_ref, "gt_id": gt_id, "class_id": p["class_id"],
                                      "features": [p["age"], p["hits"], p["last_score"], float(np.hypot(v[0], v[1])), v[0], v[1]] + _one_hot(p["class_id"]) + _geometry(box, frame_size) + [min(frames_to_exit(box, velocity=(v[0], v[1]), frame_size=frame_size), 1e4), _dis_magnitude(box, dis_points, dis_vectors[t])] + _image(gray, box)})

        # --- опорный кадр: метки и выборки q_det / q_match --------------------------
        if t in ref_of_raw:
            r = ref_of_raw[t]
            labelled = detections_with_truth_status(xyxy, conf, cls, truth[r], min_iou=DETECTION_IOU)
            gt_of = {d.detection_index: d.gt_id for d in labelled}
            occluded_of = {d.detection_index: d.occluded for d in labelled}
            crowd = _crowd_boxes(truth[r])
            ignore = np.zeros(len(xyxy), bool)
            if len(crowd) and len(xyxy):
                ignore = box_iou(xyxy, crowd).max(axis=1) >= DETECTION_IOU
            for j in np.flatnonzero(visible):
                if ignore[j]:
                    counters["det_ignored_crowd"] += 1
                    continue
                ctx = _context(xyxy[j], tracked_boxes)
                dis = _dis_prev(xyxy[j], prev_propagated)
                rows["q_det"].append([float(conf[j])] + _one_hot(int(cls[j])) + _geometry(xyxy[j], frame_size) + _image(gray, xyxy[j]) + [ctx[0], ctx[1], dis[0], dis[1]])
                meta["q_det"].append((t, r, int(j), int(cls[j]), int(gt_of.get(j) is not None), int(occluded_of.get(j, False))))
            # идентичность треков по совпадениям на этом опорном кадре — записывается ПОСЛЕ построения q_match
            visible_idx = np.flatnonzero(visible)
            cand_lists = admissible_candidates(np.asarray([pred_boxes[p["track_id"]] for p in predictions], np.float32).reshape(-1, 4), [p["class_id"] for p in predictions], xyxy[visible_idx], cls[visible_idx], distance_scale=CANDIDATE_DISTANCE) if predictions else []
            competing = Counter(j for lst in cand_lists for j in lst)
            decided = {(e["track_id"], e["det_index"]) for e in log["events"] if e["event"] in ("continued", "reactivated")}
            for p, cands in zip(predictions, cand_lists):
                identity = timeline.before(p["track_id"], r)
                if identity is None:
                    counters["match_identity_unknown"] += 1
                    continue
                if not cands:
                    counters["match_no_candidates"] += 1
                    continue
                pbox = pred_boxes[p["track_id"]]
                ious = [box_pair_geometry(pbox, xyxy[visible_idx[k]])[0] for k in cands]
                best = max(ious)
                prev_box = prev_predictions.get(p["track_id"], pbox)
                dis_box = propagate_box(prev_box, dis_points, dis_vectors, t - 1, t)[0] if t > 0 else prev_box
                v = p["velocity"]
                for k, iou in zip(cands, ious):
                    j = int(visible_idx[k])
                    _, dist, size = box_pair_geometry(pbox, xyxy[j])
                    dis_iou = box_pair_geometry(dis_box, xyxy[j])[0]
                    rows["q_match"].append([iou, dist, size, float(conf[j])] + _one_hot(p["class_id"]) + [p["age"], p["hits"], p["since_update"], float(p["state"] == "lost"), float(np.hypot(v[0], v[1])), p["last_score"], float(len(cands)), float(best - iou), float(competing[k]), dis_iou])
                    label = int(gt_of.get(j) is not None and gt_of[j] == identity)
                    meta["q_match"].append((t, r, p["track_id"], j, p["class_id"], label, int(gt_of.get(j) is None), int((p["track_id"], j) in decided)))
            for e in log["events"]:
                if e["event"] in ("continued", "reactivated", "confirmed", "created") and "det_index" in e:
                    timeline.record(ref_index=r, track_id=e["track_id"], gt_id=gt_of.get(e["det_index"]))
            # рождения, ждущие первого опорного кадра
            still = []
            for b in pending_births:
                if t - b["frame"] > BIRTH_MAX_GAP:
                    counters["birth_no_reference_frame"] += 1
                    continue
                box = output_boxes.get(b["track_id"])
                if box is None:
                    # трек ещё не активирован или уже удалён как неподтверждённый:
                    # берём рамку рождающей детекции, перенесённую DIS с кадра рождения
                    box = propagate_box(b["det_box"], dis_points, dis_vectors, b["frame"], t)[0] if t > b["frame"] else b["det_box"]
                    counters["birth_box_from_detection"] += 1
                tb, tc, tids, _ = truth_arrays(truth[r])
                gt_id = None
                if len(tb):
                    ov = box_iou(box[None, :], tb)[0]
                    ov[tc != b["class_id"]] = 0.0
                    if ov.max() >= DETECTION_IOU:
                        gt_id = tids[int(ov.argmax())]
                if len(crowd) and box_iou(box[None, :], crowd)[0].max() >= DETECTION_IOU and gt_id is None:
                    counters["birth_ignored_crowd"] += 1
                    continue
                label = birth_label(first_ref=r, gt_id=gt_id, presence=presence, horizon=BIRTH_HORIZON, n_ref=n_ref)
                if label is None:
                    counters["birth_censored"] += 1
                    continue
                rows["q_birth"].append(b["features"])
                meta["q_birth"].append((b["frame"], r, b["track_id"], b["det_index"], b["class_id"], label))
            pending_births = still
            last_ref = r
        prev_boxes = xyxy[visible]
        prev_predictions = pred_boxes
        t += 1
        if t % 300 == 0:
            print(f"[{video_id}] {t} кадров: det {len(rows['q_det'])} match {len(rows['q_match'])} birth {len(rows['q_birth'])} alive(pending) {len(pending_alive)}", flush=True)
    capture.release()
    counters["birth_pending_at_end"] += len(pending_births)
    for a in pending_alive:
        label = alive_label(lost_after_ref=a["last_ref"], gt_id=a["gt_id"], presence=presence, window=ALIVE_WINDOW, n_ref=n_ref)
        if label is None:
            counters["alive_censored"] += 1
            continue
        rows["q_alive"].append(a["features"])
        meta["q_alive"].append((a["frame"], a["last_ref"], a["track_id"], a["class_id"], label))

    output.mkdir(parents=True, exist_ok=True)
    manifest = {"video_id": video_id, "frames": t, "reference_frames": n_ref, "frame_size": frame_size, "elapsed_seconds": time.perf_counter() - started, "counters": dict(counters), "events": {}}
    names = {"q_det": DET_FEATURES, "q_match": MATCH_FEATURES, "q_birth": BIRTH_FEATURES, "q_alive": ALIVE_FEATURES}
    meta_names = {"q_det": ("frame", "ref", "det_index", "class_id", "label", "occluded"), "q_match": ("frame", "ref", "track_id", "det_index", "class_id", "label", "det_unmatched", "tracker_decided"), "q_birth": ("frame", "ref", "track_id", "det_index", "class_id", "label"), "q_alive": ("frame", "ref", "track_id", "class_id", "label")}
    for event in rows:
        features = np.asarray(rows[event], np.float32).reshape(-1, len(names[event]))
        info = np.asarray(meta[event], np.int64).reshape(-1, len(meta_names[event]))
        np.savez_compressed(output / f"{event}.npz", features=features, meta=info, feature_names=np.asarray(names[event]), meta_names=np.asarray(meta_names[event]))
        labels = info[:, list(meta_names[event]).index("label")] if len(info) else np.zeros(0, np.int64)
        classes = info[:, list(meta_names[event]).index("class_id")] if len(info) else np.zeros(0, np.int64)
        manifest["events"][event] = {"units": int(len(info)), "positives": int(labels.sum()), "positive_rate": float(labels.mean()) if len(labels) else None,
                                     "by_class": {CLASS_NAMES[c]: {"units": int((classes == c).sum()), "positives": int(labels[classes == c].sum())} for c in CLASS_IDS if (classes == c).any()}}
    save_json(output / "manifest.json", manifest)
    print(f"[OK] {video_id}: " + ", ".join(f"{k} {v['units']} ({v['positives']}+)" for k, v in manifest["events"].items()) + f", {manifest['elapsed_seconds']:.0f} с", flush=True)
    return manifest


def main() -> int:
    args = parse_args()
    manifests = {v: build_video(v, args) for v in (args.video_id or list(VIDEO_IDS))}
    summary = {"videos": manifests, "rules": {"min_conf": MIN_CONF, "detection_iou": DETECTION_IOU, "birth_horizon_ref_frames": BIRTH_HORIZON, "alive_window_ref_frames": ALIVE_WINDOW, "birth_max_gap_raw_frames": BIRTH_MAX_GAP, "candidate_distance": CANDIDATE_DISTANCE, "tracker": XYWH_CONFIG},
               "feature_names": {"q_det": DET_FEATURES, "q_match": MATCH_FEATURES, "q_birth": BIRTH_FEATURES, "q_alive": ALIVE_FEATURES}, "dis_feature_names": DIS_NAMES,
               "totals": {event: {"units": sum(m["events"][event]["units"] for m in manifests.values()), "positives": sum(m["events"][event]["positives"] for m in manifests.values())} for event in ("q_det", "q_match", "q_birth", "q_alive")},
               "counters": dict(sum((Counter(m["counters"]) for m in manifests.values()), Counter()))}
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary["totals"], ensure_ascii=False), json.dumps(summary["counters"], ensure_ascii=False))
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
