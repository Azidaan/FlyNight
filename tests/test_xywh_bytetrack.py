from __future__ import annotations

from argparse import Namespace

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import BaseTrack

from tracking.offline_association import _DetSlice, make_tracker_args
from tracking.xywh_bytetrack import XYWHByteTrack


def botsort_args() -> Namespace:
    args = make_tracker_args()
    args.gmc_method = "none"
    args.proximity_thresh = 0.5
    args.appearance_thresh = 0.8
    args.with_reid = False
    args.model = "auto"
    return args


def test_matches_botsort_without_gmc_or_reid() -> None:
    sequence = [
        (np.asarray([[10, 10, 30, 30]], np.float32), np.asarray([0.9], np.float32)),
        (np.asarray([[12, 11, 32, 31]], np.float32), np.asarray([0.8], np.float32)),
        (np.asarray([[15, 13, 36, 34]], np.float32), np.asarray([0.2], np.float32)),
        (np.asarray([[18, 15, 39, 36]], np.float32), np.asarray([0.85], np.float32)),
    ]
    def run(tracker) -> list[np.ndarray]:
        outputs = []
        for boxes, scores in sequence:
            classes = np.full(len(scores), 2, dtype=np.float32)
            outputs.append(tracker.update(_DetSlice(boxes, scores, classes)))
        return outputs

    BaseTrack.reset_id()
    expected_rows = run(BOTSORT(botsort_args()))
    BaseTrack.reset_id()
    actual_rows = run(XYWHByteTrack(make_tracker_args()))
    for actual, expected in zip(actual_rows, expected_rows):
        np.testing.assert_allclose(actual, expected, atol=1e-6)
