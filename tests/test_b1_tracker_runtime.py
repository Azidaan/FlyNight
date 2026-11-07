import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from types import SimpleNamespace
from tracking.b1_tracker_runtime import B1Tracker, CausalMasa, geometry_inputs
from tracking.offline_association import _DetSlice
from tracking.xywh_bytetrack import XYWHByteTrack
from ultralytics.trackers.basetrack import BaseTrack


def args():
    return SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.25,
                           track_buffer=30, match_thresh=.8, fuse_score=True)


def test_disabled_preserves_native_state_and_ids():
    native = XYWHByteTrack(args())
    disabled = B1Tracker(args(), disabled=True)
    BaseTrack.reset_id()
    native_count = 0
    for frame in range(16):
        boxes = np.array([[30 + frame, 30, 60 + frame, 70]], np.float32) if frame not in [6, 7] else np.zeros((0, 4), np.float32)
        det = _DetSlice(boxes, np.ones(len(boxes), np.float32) * .9, np.zeros(len(boxes), np.float32))
        BaseTrack._count = native_count
        a = native.update(det)
        native_count = BaseTrack._count
        b = disabled.update(det)
        assert np.array_equal(a, b)
        for x, y in zip(native.tracked_stracks + native.lost_stracks, disabled.tracked_stracks + disabled.lost_stracks):
            assert np.array_equal(x.mean, y.mean)
            assert np.array_equal(x.covariance, y.covariance)


def test_future_masa_image_rejected_before_model_use():
    server = CausalMasa(None)
    server.start(0, np.zeros((8, 8, 3), np.uint8), np.zeros((0, 4)), np.zeros(0))
    import pytest
    with pytest.raises(AssertionError):
        server.embed(1, [[0, 0, 4, 4]])


def test_real_context_keeps_original_assignment_column():
    from tracking.z13_readout import HexGeometry
    geom = HexGeometry(np.array([[0, 0], [5, 0], [0, 5], [5, 5]], np.float32), 10)
    det = [SimpleNamespace(xyxy=np.array([0, 0, 9, 9]), score=.9, cls=1)]
    choices, masks, geo = geometry_inputs(geom, [0, 0, 10, 10], [5, 5], det, 1,
                                         np.array([[.2], [.1]]), 0, np.array([-1, 0]))
    assert choices == [-1, 0]
    assert geo.shape == (2, 14)
    assert geo[0, -1] == 1 and geo[1, -1] == 0
    assert masks.shape == (2, 4)


def test_live_base_first_query_matches_saved_frozen_features():
    from pathlib import Path
    from tracking.b1_tracker_runtime import FrozenBase
    root = Path(__file__).resolve().parents[1]
    folder = root / 'runs/b1_real_inputs_20261026'
    with np.load(folder / 'base_inputs.npz') as z:
        i = int(np.flatnonzero(z['uid'] == 0)[0]); video = str(z['name'][i])
        count = int(z['mask'][i].sum()); geo = z['geo'][i, :count].copy(); expected = z['F1'][i].copy()
    with np.load(root / f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz') as z:
        j = int(np.flatnonzero(z['uid'] == 0)[0]); region = z['region'][j, :count - 1].astype(np.float32); coarse = z['exemplar'][j].astype(np.float32)
    with np.load(root / f'runs/z13_masa_exemplar_snap_20261026/videos/{video}.npz') as z:
        j = int(np.flatnonzero(z['uid'] == 0)[0]); snapped = z['exemplar'][j].astype(np.float32)
    base = FrozenBase()
    p, _, _, _, features = base.probability(geo, region, coarse, snapped, None, 0)
    assert np.array_equal(np.isnan(features[:count]), np.isnan(expected[:count]))
    assert np.allclose(features[:count], expected[:count], atol=1e-5, rtol=0, equal_nan=True)
    with np.load(folder / 'base_predictions.npz') as z:
        # uid0 belongs to fit20, hence full-fit fold4, as at live runtime.
        assert np.array_equal(p, z['identity_probability'][i], equal_nan=True)
