"""Диагностика наличия кандидатов и реальных выходных траекторий.

Покадровое сопоставление здесь — максимум числа пар, затем суммы IoU.
Это диагностическая разметка событий, не замена официальным HOTA/CLEAR.
Crowd исключается из GT; непарные рамки с IoA > .5 в crowd игнорируются.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from datasets.bdd100k_loader import CLASS_NAMES

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


def overlaps(a, b, *, ioa=False):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(axis=2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
    bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)[None, :]
    return inter / np.maximum(aa if ioa else aa + bb - inter, 1e-12)


def match_one_to_one(boxes, classes, gt_boxes, gt_classes, threshold=.5):
    """Вернуть индекс GT для каждой рамки, -1 при отсутствии пары."""
    result = np.full(len(boxes), -1, dtype=int)
    if not len(boxes) or not len(gt_boxes):
        return result
    iou = overlaps(boxes, gt_boxes)
    valid = (iou >= threshold) & (np.asarray(classes)[:, None]
                                  == np.asarray(gt_classes)[None, :])
    # Бонус обеспечивает максимальную мощность допустимого сопоставления.
    score = np.where(valid, min(iou.shape) + 1 + iou, 0)
    rows, cols = linear_sum_assignment(-score)
    ok = valid[rows, cols]
    result[rows[ok]] = cols[ok]
    return result


@dataclass
class TruthFrame:
    ids: list[str]
    boxes: np.ndarray
    classes: np.ndarray
    ages: np.ndarray
    occluded: np.ndarray
    crowd: np.ndarray
    name: str = ""


def read_truth(path: Path):
    frames = json.loads(path.read_text(encoding="utf-8"))
    out, first, observations = {}, {}, Counter()
    for frame in sorted(frames, key=lambda x: x["frameIndex"]):
        fi = int(frame["frameIndex"])
        ids, boxes, classes, ages, occluded, crowd = [], [], [], [], [], []
        for label in frame.get("labels") or []:
            box = label.get("box2d")
            if not box:
                continue
            b = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            attrs = label.get("attributes") or {}
            if attrs.get("crowd", False):
                crowd.append(b)
                continue
            if label["category"] not in CLASS_IDS:
                continue
            oid = str(label["id"])
            if oid in ids:
                raise ValueError(f"{path.name}/{fi}: duplicate GT ID {oid}")
            first.setdefault(oid, fi)
            observations[oid] += 1
            ids.append(oid)
            boxes.append(b)
            classes.append(CLASS_IDS[label["category"]])
            ages.append(fi - first[oid])
            occluded.append(bool(attrs.get("occluded", False)))
        out[fi] = TruthFrame(ids, np.asarray(boxes).reshape(-1, 4),
                             np.asarray(classes, dtype=int), np.asarray(ages, dtype=int),
                             np.asarray(occluded, dtype=bool),
                             np.asarray(crowd).reshape(-1, 4), frame.get("name", ""))
    return out, first, observations


def ignored_unmatched(boxes, match, crowd):
    return (match < 0) & np.any(overlaps(boxes, crowd, ioa=True) > .5, axis=1)


def label_candidates(boxes, classes, scores, truth, high=.25):
    """Сначала сопоставить высокие баллы; низкие не отбирают у них GT."""
    mapping = np.full(len(boxes), -1, dtype=int)
    high_ids = np.flatnonzero(scores >= high)
    mapping[high_ids] = match_one_to_one(
        boxes[high_ids], classes[high_ids], truth.boxes, truth.classes)
    used = set(mapping[mapping >= 0].tolist())
    remaining = np.asarray([i for i in range(len(truth.ids)) if i not in used], dtype=int)
    low_ids = np.flatnonzero(scores < high)
    low_match = match_one_to_one(boxes[low_ids], classes[low_ids],
                                 truth.boxes[remaining], truth.classes[remaining])
    ok = low_match >= 0
    mapping[low_ids[ok]] = remaining[low_match[ok]]
    return mapping


class LifecycleAudit:
    """Первый правильный ВЫХОД трекера и события, а не первый отклик детектора."""

    def __init__(self, first):
        self.first = first
        self.last_correct = {}
        self.first_correct = {}
        self.last_identity = {}
        self.seen_output_ids = set()
        self.counts = Counter()
        self.events = []

    def observe(self, fi, output, truth):
        out = np.asarray(output, dtype=float).reshape(-1, 8)
        matched = match_one_to_one(out[:, :4], out[:, 6], truth.boxes, truth.classes)
        ignored = ignored_unmatched(out[:, :4], matched, truth.crowd)
        self.counts["frames"] += 1
        self.counts["gt_observations"] += len(truth.ids)
        self.counts["output_tp"] += int((matched >= 0).sum())
        self.counts["output_fp"] += int(((matched < 0) & ~ignored).sum())
        self.counts["output_ignored"] += int(ignored.sum())
        self.counts["output_fn"] += len(truth.ids) - int((matched >= 0).sum())
        for i, row in enumerate(out):
            tid = int(row[4])
            is_new = tid not in self.seen_output_ids
            self.seen_output_ids.add(tid)
            gt_idx = matched[i]
            oid = truth.ids[gt_idx] if gt_idx >= 0 else None
            event = "continued_output"
            if is_new:
                event = ("ignored_birth" if ignored[i] else
                         "false_birth" if oid is None else
                         "duplicate_birth" if oid in self.first_correct else "true_birth")
                self.counts[event] += 1
            if oid is not None:
                if oid in self.last_identity and self.last_identity[oid] != tid:
                    self.counts["audit_id_switches"] += 1
                    self.events.append({"frame": fi, "event": "audit_id_switch",
                                        "track_id": tid, "gt_id": oid})
                self.first_correct.setdefault(oid, fi)
                self.last_correct[oid] = fi
                self.last_identity[oid] = tid
            self.events.append({"frame": fi, "event": event, "track_id": tid,
                                "gt_id": oid, "ignored": bool(ignored[i])})
        return matched

    def summary(self):
        out = dict(self.counts)
        out["gt_tracks"] = len(self.first)
        out["never_tracked"] = len(self.first) - len(self.first_correct)
        delays = [frame - self.first[oid] for oid, frame in self.first_correct.items()]
        out["first_output_delay_mean_found_only"] = float(np.mean(delays)) if delays else None
        out["first_output_delay_sum_found_only"] = int(sum(delays))
        for limit in (0, 1, 3, 5):
            out[f"tracked_within_{limit}_all_gt"] = sum(d <= limit for d in delays)
        return out
