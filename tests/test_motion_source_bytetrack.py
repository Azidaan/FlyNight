"""MotionSourceByteTrack: режим по умолчанию равен стенду XYWHByteTrack; буфер и DIS меняют ассоциацию."""

import numpy as np
import pytest

from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache
from tracking.motion_source_bytetrack import MotionSourceByteTrack, buffered, iou_matrix
from tracking.offline_association import _DetSlice, make_tracker_args
from tracking.xywh_bytetrack import XYWHByteTrack
from pathlib import Path

CACHE = Path("runs/video_rate/b1c81faa-3df17267/detections/raw_video.npz")
XYWH = dict(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=30, match_thresh=0.8)


def _run(tracker, detections, frames):
    out = []
    for f in range(frames):
        xyxy, conf, cls = detections_for_frame(detections, f)
        out.append(np.asarray(tracker.update(_DetSlice(xyxy, conf, cls))).copy())
    return out


@pytest.mark.skipif(not CACHE.exists(), reason="нет кэша детекций val")
def test_default_is_byte_identical_to_bench():
    det = load_detection_cache(CACHE)
    a = _run(XYWHByteTrack(make_tracker_args(**XYWH)), det, 300)
    b = _run(MotionSourceByteTrack(make_tracker_args(**XYWH)), det, 300)
    assert all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not CACHE.exists(), reason="нет кэша детекций val")
def test_zero_dis_motion_changes_only_via_center():
    det = load_detection_cache(CACHE)
    t = MotionSourceByteTrack(make_tracker_args(**XYWH), motion="dis", dis_propagate=lambda box, raw: np.asarray(box, float))
    _run(t, det, 100)
    assert t.stats["switched"] > 0


def test_buffered_iou_grows_overlap():
    a = np.array([[0, 0, 10, 10]], float); b = np.array([[12, 0, 22, 10]], float)
    assert iou_matrix(a, b)[0, 0] == 0.0
    assert iou_matrix(buffered(a, 0.2), buffered(b, 0.2))[0, 0] > 0.0


def test_switch_requires_head():
    with pytest.raises(ValueError):
        MotionSourceByteTrack(make_tracker_args(**XYWH), motion="switch", dis_propagate=lambda b, r: b)


def test_adaptive_buffer_keeps_perfect_pair_at_one():
    """Совпадающие рамки трека и детекции должны давать IoU = 1 при любом буфере трека."""
    from types import SimpleNamespace
    t = MotionSourceByteTrack(make_tracker_args(**XYWH), buffer=0.2, buffer_gain=1.0, reliability=lambda a, b, c: np.ones(len(a)))
    trk = SimpleNamespace(xyxy=np.array([10, 10, 50, 90.0])); det = SimpleNamespace(xyxy=np.array([10, 10, 50, 90.0]), score=1.0)
    t._p_bad = {id(trk): 0.9}
    t.args.fuse_score = False
    assert t.get_dists([trk], [det])[0, 0] == pytest.approx(0.0)
