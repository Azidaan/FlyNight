"""Проверки протокола: уникальность пар, реальные рождения, изоляция вмешательств."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT

from evaluation.tracker_lifecycle import (LifecycleAudit, TruthFrame, ignored_unmatched,
                                         label_candidates, match_one_to_one, read_truth)
from scripts.tracker_zoo import load_args
from scripts.confidence_rescue import chosen_fraction, select_cohorts
from tracking.confidence_audit import AuditedBoTSORT, IndexedDetections


def args():
    return load_args("botsort", {"gmc_method": "none", "with_reid": False})


def det(scores, boxes=None):
    if boxes is None:
        boxes = [[10, 10, 50, 50] for _ in scores]
    return IndexedDetections(np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                             np.asarray(scores, dtype=np.float32),
                             np.zeros(len(scores), dtype=int))


def truth():
    return TruthFrame(["a"], np.array([[10, 10, 50, 50]]), np.array([0]),
                      np.array([0]), np.array([False]), np.empty((0, 4)))


class MatchingTest(unittest.TestCase):
    def test_detection_cannot_explain_two_objects(self):
        m = match_one_to_one([[0, 0, 10, 10]], [0],
                             [[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0])
        self.assertEqual(int((m >= 0).sum()), 1)

    def test_wrong_class_and_empty(self):
        self.assertEqual(match_one_to_one([[0, 0, 10, 10]], [1],
                                         [[0, 0, 10, 10]], [0]).tolist(), [-1])
        self.assertEqual(match_one_to_one([], [], [], []).tolist(), [])

    def test_high_candidate_is_not_stolen_by_low_duplicate(self):
        x = det([.9, .2])
        self.assertEqual(label_candidates(x.xyxy, x.cls, x.conf, truth()).tolist(), [0, -1])

    def test_crowd_does_not_remove_a_matched_object(self):
        b = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(ignored_unmatched(b, np.array([0, -1]), b[:1]).tolist(), [False, True])

    def test_age_is_frame_distance_and_crowd_is_excluded(self):
        box = dict(x1=0, y1=0, x2=10, y2=10)
        label = dict(id="x", category="car", box2d=box, attributes={})
        frames = [{"frameIndex": 2, "labels": [label]},
                  {"frameIndex": 7, "labels": [label, dict(label, id="crowd",
                                                        attributes={"crowd": True})]}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "truth.json"
            p.write_text(json.dumps(frames))
            result, first, _ = read_truth(p)
        self.assertEqual(first, {"x": 2})
        self.assertEqual(result[7].ages.tolist(), [5])
        self.assertEqual(result[7].ids, ["x"])


class TrackerTest(unittest.TestCase):
    def test_installed_second_stage_rejects_perfect_low_detection(self):
        t = AuditedBoTSORT(args())
        self.assertEqual(len(t.update(det([.9]))), 1)
        self.assertEqual(len(t.update(det([.2]))), 0)

    def test_iou_control_keeps_id_and_original_score(self):
        t = AuditedBoTSORT(args(), second_iou_only=True)
        first = t.update(det([.9]))
        second = t.update(det([.2]))
        self.assertEqual(second[0, 4], first[0, 4])
        self.assertAlmostEqual(float(second[0, 5]), .2)
        self.assertTrue(t.args.fuse_score)
        self.assertTrue(any(e.get("stage") == "second" for e in t.events))

    def test_control_does_not_change_first_stage_cost(self):
        t = AuditedBoTSORT(args(), second_iou_only=True)
        t.update(det([.9]))
        candidates = t.init_track(det([.4]))
        distance = t.get_dists(t.tracked_stracks, candidates)
        self.assertAlmostEqual(float(distance[0, 0]), .6, places=6)

    def test_no_intervention_matches_unwrapped_tracker(self):
        frames = [det([.9]), det([.8]), det([.2]), det([]), det([.95])]
        native = BOTSORT(args())
        expected = [native.update(x) for x in frames]
        audited = AuditedBoTSORT(args())
        actual = [audited.update(x) for x in frames]
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_gate_creates_tentative_then_actual_confirmation(self):
        t = AuditedBoTSORT(args(), second_iou_only=True)
        t.update(det([]))
        x = det([.2])
        self.assertEqual(len(t.update(x, allowed=[0])), 0)
        self.assertTrue(any(e["event"] == "created" and not e["confirmed"] for e in t.events))
        self.assertAlmostEqual(float(x.conf[0]), .2)
        output = t.update(det([.9]))
        self.assertEqual(len(output), 1)
        self.assertTrue(any(e["event"] == "confirmed" for e in t.events))

    def test_original_detection_index_survives_splitting(self):
        t = AuditedBoTSORT(args())
        x = det([.02, .9], [[200, 200, 210, 210], [10, 10, 50, 50]])
        output = t.update(x)
        self.assertEqual(int(output[0, 7]), 1)
        created = [e for e in t.events if e["event"] == "created"]
        self.assertEqual(created[0]["det_index"], 1)


class OutputAuditTest(unittest.TestCase):
    def test_delay_uses_output_and_includes_never_tracked_in_denominator(self):
        audit = LifecycleAudit({"a": 0, "b": 0})
        audit.observe(0, np.empty((0, 8)), truth())
        audit.observe(1, np.array([[10, 10, 50, 50, 7, .9, 0, 0]]), truth())
        summary = audit.summary()
        self.assertEqual(summary["never_tracked"], 1)
        self.assertEqual(summary["first_output_delay_mean_found_only"], 1.)
        self.assertEqual(summary["tracked_within_0_all_gt"], 0)
        self.assertEqual(summary["tracked_within_1_all_gt"], 1)


class CohortTest(unittest.TestCase):
    def test_equal_counts_strata_and_left_censoring(self):
        def row(frame, index, age, cls=0):
            return dict(frame=frame, index=index, age=age, **{"class": cls},
                        score=.2, area=100, occluded=True, context="untracked")
        rows = [row(0, 0, 0), row(5, 1, 0), row(6, 2, 0, 1),
                row(20, 0, 15), row(21, 1, 16)]
        groups, counts = select_cohorts(rows, 42)
        self.assertEqual(groups["birth"], [(5, 1)])
        self.assertEqual(len(groups["mature"]), 1)
        self.assertEqual(groups["initial"], [(0, 0)])
        self.assertEqual(counts["birth_available"], 2)

    def test_fraction_sets_are_nested_and_never_round_up(self):
        sequence = [(i, 0) for i in range(7)]
        self.assertEqual(len(chosen_fraction(sequence, .25)), 1)
        self.assertTrue(set(chosen_fraction(sequence, .25)) <= set(chosen_fraction(sequence, .5)))
        self.assertEqual(chosen_fraction(sequence, 1), sequence)


if __name__ == "__main__":
    unittest.main()
