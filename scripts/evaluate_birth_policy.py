"""Вето рождений по q_birth в XYWHByteTrack: обучение головы, выбор τ на train, оценка на held-out.

Карточка birth_policy_qbirth_v1 (цель 6). Контроль — τ = 0 (побайтово стенд).
Отложенные видео используются только для итоговых метрик.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics.trackers.basetrack import BaseTrack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.build_event_dataset import BIRTH_HORIZON, BIRTH_MAX_GAP, CLASS_IDS, DETECTION_IOU, MIN_CONF, VIDEO_IDS, _crowd_boxes, _xywh_to_xyxy  # noqa: E402
from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS, propagate_box, truth_arrays  # noqa: E402
from scripts.freeze_bench import XYWH_CONFIG  # noqa: E402
from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, load_motion_cache, save_json, scalabel_frame  # noqa: E402
from tracking.birth_policy import BirthHead, DeferredBirthByteTrack, QBirthByteTrack  # noqa: E402
from tracking.decision_log import IndexedDetections  # noqa: E402
from tracking.event_dataset import birth_label  # noqa: E402
from tracking.flyvis_tracker import box_iou  # noqa: E402
from tracking.offline_association import make_tracker_args  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "runs" / "event_dataset")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "birth_policy")
    parser.add_argument("--precision-margin", type=float, default=0.05)
    parser.add_argument("--mode", choices=("veto", "deferred"), default="veto")
    parser.add_argument("--head-json", type=Path, default=None, help="deferred: голова и τ из прежнего прогона (runs/birth_policy)")
    parser.add_argument("--confirm-hits", nargs="+", type=int, default=(2, 3, 5))
    return parser.parse_args()


def train_head(dataset: Path) -> tuple[BirthHead, dict]:
    features, labels = [], []
    names = None
    for video_id in TRAIN_VIDEO_IDS:
        with np.load(dataset / video_id / "q_birth.npz") as data:
            names = [str(n) for n in data["feature_names"]]
            features.append(data["features"]); labels.append(data["meta"][:, list(data["meta_names"]).index("label")])
    X, y = np.concatenate(features), np.concatenate(labels).astype(np.int8)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(X, y)
    head = BirthHead.from_pipeline(pipeline, names)
    q = pipeline.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, q)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    tau_f1 = float(thresholds[int(np.argmax(f1))])
    control_precision = float(y.mean())  # контроль рождает всех кандидатов
    target = control_precision + 0.05
    ok = np.flatnonzero(precision[:-1] >= target)
    tau_prec = float(thresholds[ok.min()]) if len(ok) else 1.0
    info = {"train_units": int(len(y)), "train_positives": int(y.sum()), "control_precision_train": control_precision, "tau_f1": tau_f1, "tau_prec": tau_prec, "precision_target": target,
            "recall_at_tau_prec_train": float(recall[:-1][ok.min()]) if len(ok) else 0.0}
    return head, info


def make_tracker(spec: dict, head: BirthHead):
    if spec.get("kind") == "deferred":
        return DeferredBirthByteTrack(make_tracker_args(**XYWH_CONFIG), head=head, tau=spec["tau"], confirm_hits=spec["confirm_hits"], class_ids=CLASS_IDS, defer_all=spec.get("defer_all", False))
    return QBirthByteTrack(make_tracker_args(**XYWH_CONFIG), head=head, tau=spec["tau"], class_ids=CLASS_IDS)


def run_variant(name: str, spec: dict, head: BirthHead, args: argparse.Namespace, video_ids: list[str], result_dir: Path) -> dict:
    per_video = {}
    for video_id in video_ids:
        detections = load_detection_cache(args.root / video_id / "detections" / "raw_video.npz")
        matches = json.loads((args.root / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
        ref_of_raw = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
        truth = json.loads((args.labels / f"{video_id}.json").read_text(encoding="utf-8"))
        n_ref = len(truth)
        dis_points, dis_vectors = load_motion_cache(args.root / video_id / "motion" / "dis_quarter.npz", 1.0)
        presence: dict[str, set[int]] = defaultdict(set)
        first_seen: dict[str, int] = {}
        for r, frame in enumerate(truth):
            for gt_id in truth_arrays(frame)[2]:
                presence[gt_id].add(r); first_seen.setdefault(gt_id, r)
        capture = cv2.VideoCapture(str(args.videos / f"{video_id}.mov"))
        frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        BaseTrack.reset_id()
        tracker = make_tracker(spec, head)
        rows, pending, counters = [], [], Counter()
        seen_ids: set[int] = set()
        births_out: list[tuple[int, int]] = []
        first_correct: dict[str, int] = {}
        prev_boxes = np.zeros((0, 4), np.float32)
        timings = []
        t = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            xyxy, conf, cls = detections_for_frame(detections, t)
            prev_propagated = np.asarray([propagate_box(b, dis_points, dis_vectors, t - 1, t)[0] for b in prev_boxes], np.float32).reshape(-1, 4) if t > 0 and len(prev_boxes) else np.zeros((0, 4), np.float32)
            tracker.set_frame_context(gray=gray, prev_propagated=prev_propagated, frame_size=frame_size)
            tick = time.perf_counter()
            log = tracker.update_with_log(IndexedDetections(xyxy, conf, cls))
            timings.append(time.perf_counter() - tick)
            output = log["output"]
            # рождение = первое появление ID в выходе (опубликованный новый ID), одинаково для всех вариантов
            for row in output:
                tid = int(row[4])
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    pending.append({"track_id": tid, "frame": t, "det_box": row[:4].astype(np.float32).copy(), "class_id": int(row[6])})
            if t in ref_of_raw:
                r = ref_of_raw[t]
                rows.append(scalabel_frame(video_id, truth[r], output))
                output_boxes = {int(row[4]): row[:4].astype(np.float32) for row in output}
                tb, tc, tids, _ = truth_arrays(truth[r]); crowd = _crowd_boxes(truth[r])
                # первый правильный выход каждого GT-объекта
                if len(tb) and len(output):
                    ov = box_iou(np.asarray([row[:4] for row in output], np.float32), tb)
                    for k, gt_id in enumerate(tids):
                        if gt_id not in first_correct and ov[:, k].max() >= DETECTION_IOU:
                            first_correct[gt_id] = r
                still = []
                for b in pending:
                    if t - b["frame"] > BIRTH_MAX_GAP:
                        counters["birth_no_reference_frame"] += 1; continue
                    box = output_boxes.get(b["track_id"])
                    if box is None:
                        box = propagate_box(b["det_box"], dis_points, dis_vectors, b["frame"], t)[0] if t > b["frame"] else b["det_box"]
                    gt_id = None
                    if len(tb):
                        o = box_iou(box[None, :], tb)[0]; o[tc != b["class_id"]] = 0.0
                        if o.max() >= DETECTION_IOU:
                            gt_id = tids[int(o.argmax())]
                    if gt_id is None and len(crowd) and box_iou(box[None, :], crowd)[0].max() >= DETECTION_IOU:
                        counters["birth_ignored_crowd"] += 1; continue
                    label = birth_label(first_ref=r, gt_id=gt_id, presence=presence, horizon=BIRTH_HORIZON, n_ref=n_ref)
                    if label is None:
                        counters["birth_censored"] += 1; continue
                    births_out.append((b["frame"], label))
                pending = still
            prev_boxes = xyxy[conf > MIN_CONF]
            t += 1
        capture.release()
        result_dir.mkdir(parents=True, exist_ok=True)
        save_json(result_dir / f"{video_id}.json", rows)
        labels = np.asarray([lab for _, lab in births_out], np.int8)
        delays = [first_correct[g] - first_seen[g] for g in first_seen if g in first_correct and len(presence[g]) >= BIRTH_HORIZON]
        found = [g for g in first_seen if len(presence[g]) >= BIRTH_HORIZON]
        per_video[video_id] = {
            "births": int(len(labels)), "true_births": int(labels.sum()), "false_births": int((labels == 0).sum()), "birth_precision": float(labels.mean()) if len(labels) else None,
            "candidates_seen": int(len(tracker.birth_log)), "vetoed": int(sum(e["vetoed"] for e in tracker.birth_log)), "deferred": int(sum(e.get("deferred", False) for e in tracker.birth_log)),
            "objects_eligible": int(len(found)), "objects_found": int(len(delays)), "mean_first_output_delay_ref": float(np.mean(delays)) if delays else None,
            "counters": dict(counters), "association_ms_mean": float(np.mean(timings) * 1000),
        }
        print(f"[{name}] {video_id}: births {len(labels)} ({int(labels.sum())} true, {int((labels == 0).sum())} false), vetoed {per_video[video_id]['vetoed']}/{len(tracker.birth_log)}", flush=True)
    return {"per_video": per_video}


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "deferred" and args.head_json is not None:
        head = BirthHead.from_json(args.head_json)
        head_info = json.loads((args.head_json.parent / "report.json").read_text(encoding="utf-8"))["head"]
        head_info["reused_from"] = str(args.head_json)
    else:
        head, head_info = train_head(args.dataset)
        save_json(args.output / "head.json", head.to_json())
    print(f"[HEAD] {head_info}", flush=True)
    if args.mode == "deferred":
        variants = {"control": {"kind": "veto", "tau": 0.0}}
        for n in args.confirm_hits:
            variants[f"deferred_n{n}"] = {"kind": "deferred", "tau": head_info["tau_f1"], "confirm_hits": n}
        variants["defer_all_n3"] = {"kind": "deferred", "tau": 1.0, "confirm_hits": 3, "defer_all": True}
    else:
        variants = {"control": {"kind": "veto", "tau": 0.0}, "tau_f1": {"kind": "veto", "tau": head_info["tau_f1"]}, "tau_prec": {"kind": "veto", "tau": head_info["tau_prec"]}}
    video_ids = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    report = {"protocol": {"head": "StandardScaler + LogisticRegression(C=1) на q_birth четырёх обучающих видео (base + DIS)", "tau_rule": "τ_F1 — максимум F1 на train; τ_prec — минимальное τ с точностью рождений ≥ контроль + 0,05 на train", "tracker": XYWH_CONFIG, "train_video_ids": list(TRAIN_VIDEO_IDS), "test_video_ids": list(TEST_VIDEO_IDS)}, "head": head_info, "variants": {}}
    for name, spec in variants.items():
        print(f"[RUN] {name} {spec}", flush=True)
        result_dir = args.output / name
        run = run_variant(name, spec, head, args, video_ids, result_dir)
        run["tau"] = spec["tau"]; run["spec"] = spec
        run["metrics_test"] = evaluate_hota(result_dir, list(TEST_VIDEO_IDS))
        run["metrics_train"] = evaluate_hota(result_dir, list(TRAIN_VIDEO_IDS))
        run["metrics_test_by_video"] = {v: evaluate_hota(result_dir, [v]) for v in TEST_VIDEO_IDS}
        report["variants"][name] = run
        m = run["metrics_test"]
        print(f"[{name}] TEST HOTA {m['HOTA']:.4f} AssA {m['AssA']:.4f} IDF1 {m['IDF1']:.4f} IDSw {m['IDSw']} FP {m['CLR_FP']} FN {m['CLR_FN']} | false births {sum(v['false_births'] for k, v in run['per_video'].items() if k in TEST_VIDEO_IDS)}", flush=True)
    keys = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSw", "CLR_FP", "CLR_FN", "Frag")
    ctrl = report["variants"]["control"]
    summary = {}
    for name in [k for k in report["variants"] if k != "control"]:
        var = report["variants"][name]
        summary[name] = {
            "delta_test": {k: var["metrics_test"][k] - ctrl["metrics_test"][k] for k in keys},
            "false_births_test": {v: (ctrl["per_video"][v]["false_births"], var["per_video"][v]["false_births"]) for v in TEST_VIDEO_IDS},
            "true_births_test": {v: (ctrl["per_video"][v]["true_births"], var["per_video"][v]["true_births"]) for v in TEST_VIDEO_IDS},
            "delay_test": {v: (ctrl["per_video"][v]["mean_first_output_delay_ref"], var["per_video"][v]["mean_first_output_delay_ref"]) for v in TEST_VIDEO_IDS},
            "per_video_delta": {v: {k: var["metrics_test_by_video"][v][k] - ctrl["metrics_test_by_video"][v][k] for k in ("HOTA", "IDF1", "IDSw", "CLR_FP", "CLR_FN")} for v in TEST_VIDEO_IDS},
        }
        s = summary[name]
        s["acceptance"] = {
            "false_births_lower_every_video": all(b < a for a, b in s["false_births_test"].values()),
            "fp_lower_every_video": all(d["CLR_FP"] < 0 for d in s["per_video_delta"].values()),
            "idsw_not_higher_every_video": all(d["IDSw"] <= 0 for d in s["per_video_delta"].values()),
            "hota_idf1_not_below_control_minus_0_1": s["delta_test"]["HOTA"] >= -0.1 and s["delta_test"]["IDF1"] >= -0.1,
        }
    if args.mode == "deferred":
        # выбор N по обучающим видео: максимум HOTA при ложных рождениях ниже контроля
        ctrl_fb_train = sum(ctrl["per_video"][v]["false_births"] for v in TRAIN_VIDEO_IDS)
        eligible = {n: report["variants"][n]["metrics_train"]["HOTA"] for n in summary if n.startswith("deferred") and sum(report["variants"][n]["per_video"][v]["false_births"] for v in TRAIN_VIDEO_IDS) < ctrl_fb_train}
        report["selected_on_train"] = max(eligible, key=eligible.get) if eligible else None
        report["train_hota"] = {n: report["variants"][n]["metrics_train"]["HOTA"] for n in report["variants"]}
    report["summary"] = summary
    save_json(args.output / "report.json", report)
    print(json.dumps({k: {"delta_test": {kk: round(vv, 4) for kk, vv in v["delta_test"].items()}, "acceptance": v["acceptance"]} for k, v in summary.items()}, ensure_ascii=False, indent=1))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
