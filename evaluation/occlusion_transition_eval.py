"""Диагностика сохранения ID между размеченными кадрами вокруг окклюзии."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.flyvis_tracker import box_iou


def frame_objects(frame: dict, *, truth: bool) -> list[dict]:
    rows = []
    for label in frame.get("labels") or []:
        box = label.get("box2d")
        if box is None:
            continue
        attributes = label.get("attributes") or {}
        rows.append(
            {
                "id": str(label["id"]),
                "category": label["category"],
                "box": np.asarray(
                    [box["x1"], box["y1"], box["x2"], box["y2"]],
                    dtype=np.float32,
                ),
                "occluded": bool(attributes.get("occluded", False)) if truth else False,
            }
        )
    return rows


def match_frame(truth_frame: dict, result_frame: dict, iou_threshold: float = 0.5) -> dict:
    """Сопоставить GT с выводом трекера по классу и IoU."""
    truth = frame_objects(truth_frame, truth=True)
    result = frame_objects(result_frame, truth=False)
    mapping = {}
    categories = sorted({row["category"] for row in truth})
    for category in categories:
        gt_indices = [i for i, row in enumerate(truth) if row["category"] == category]
        pr_indices = [i for i, row in enumerate(result) if row["category"] == category]
        if not gt_indices or not pr_indices:
            continue
        gt_boxes = np.stack([truth[i]["box"] for i in gt_indices])
        pr_boxes = np.stack([result[i]["box"] for i in pr_indices])
        overlaps = box_iou(gt_boxes, pr_boxes)
        rows, columns = linear_sum_assignment(1.0 - overlaps)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if overlaps[row, column] < iou_threshold:
                continue
            gt = truth[gt_indices[row]]
            prediction = result[pr_indices[column]]
            mapping[gt["id"]] = {
                "prediction_id": prediction["id"],
                "occluded": gt["occluded"],
                "iou": float(overlaps[row, column]),
            }
    return mapping


def evaluate_occlusion_transitions(
    truth: list[dict], result: list[dict], iou_threshold: float = 0.5
) -> dict:
    if len(truth) != len(result):
        raise ValueError("Число кадров GT и результата должно совпадать")
    matches = [
        match_frame(gt_frame, result_frame, iou_threshold)
        for gt_frame, result_frame in zip(truth, result)
    ]
    truth_rows = [frame_objects(frame, truth=True) for frame in truth]
    counters = {
        "touches_occlusion": {"eligible": 0, "matched_both": 0, "retained": 0, "switched": 0},
        "visible_visible": {"eligible": 0, "matched_both": 0, "retained": 0, "switched": 0},
    }
    for index in range(1, len(truth)):
        previous_truth = {row["id"]: row for row in truth_rows[index - 1]}
        current_truth = {row["id"]: row for row in truth_rows[index]}
        for gt_id in previous_truth.keys() & current_truth.keys():
            group = (
                "touches_occlusion"
                if previous_truth[gt_id]["occluded"] or current_truth[gt_id]["occluded"]
                else "visible_visible"
            )
            counter = counters[group]
            counter["eligible"] += 1
            before = matches[index - 1].get(gt_id)
            after = matches[index].get(gt_id)
            if before is None or after is None:
                continue
            counter["matched_both"] += 1
            if before["prediction_id"] == after["prediction_id"]:
                counter["retained"] += 1
            else:
                counter["switched"] += 1
    for counter in counters.values():
        counter["retention_given_matched"] = (
            counter["retained"] / counter["matched_both"]
            if counter["matched_both"]
            else None
        )
        counter["retained_over_eligible"] = (
            counter["retained"] / counter["eligible"]
            if counter["eligible"]
            else None
        )
    return counters


def evaluate_result_directory(
    result_dir: str | Path,
    video_ids: list[str],
    *,
    truth_dir: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    result_dir = Path(result_dir)
    truth_dir = Path(truth_dir)
    per_video = {}
    combined = {
        "touches_occlusion": {"eligible": 0, "matched_both": 0, "retained": 0, "switched": 0},
        "visible_visible": {"eligible": 0, "matched_both": 0, "retained": 0, "switched": 0},
    }
    for video_id in video_ids:
        truth = json.loads((truth_dir / f"{video_id}.json").read_text(encoding="utf-8"))
        result = json.loads((result_dir / f"{video_id}.json").read_text(encoding="utf-8"))
        metrics = evaluate_occlusion_transitions(truth, result, iou_threshold)
        per_video[video_id] = metrics
        for group in combined:
            for key in ("eligible", "matched_both", "retained", "switched"):
                combined[group][key] += metrics[group][key]
    for counter in combined.values():
        counter["retention_given_matched"] = (
            counter["retained"] / counter["matched_both"]
            if counter["matched_both"]
            else None
        )
        counter["retained_over_eligible"] = (
            counter["retained"] / counter["eligible"]
            if counter["eligible"]
            else None
        )
    return {"combined": combined, "per_video": per_video, "iou_threshold": iou_threshold}
