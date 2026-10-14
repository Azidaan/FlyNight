from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from tracking.z13_readout import HexGeometry
from tracking.z13_spatial_localizer import SpatialLocalizer, percentile_threshold, training_integration


def test_unsupervised_percentile_retains_all_ties():
    assert percentile_threshold([.9, .8, .8, .1], .5) == {
        'tau': .8, 'target_rank': 2, 'selected': 3, 'total': 4}
    with pytest.raises(ValueError):
        percentile_threshold([float('nan')])


def test_neuron_timestep_matches_training_not_video_rate():
    steps, dt = training_integration()
    assert steps == 2 and dt == pytest.approx(1 / 60)
    assert steps * dt == pytest.approx(1 / 30)


def localizer(tau):
    loc = SpatialLocalizer.__new__(SpatialLocalizer)
    points = np.stack(np.meshgrid(np.arange(31), np.arange(24)), -1).reshape(-1, 2)[:721]
    loc.geom = HexGeometry(points.astype(np.float32) * 3 + 5, 100)
    loc.torch, loc.arm, loc.dev, loc.tau = torch, 'B', 'cpu', tau
    loc.frame_size, loc.max_batch = (1280, 720), 4
    loc.hist, loc.eyes = {}, {}
    loc.stats = {'steps': 0, 'births': 0, 'interventions': 0, 'refusals': 0, 'nonfinite_refusals': 0}
    class Backbone:
        def init(self, x):
            return {'h': torch.zeros(len(x), 1, 721)}
        def step(self, state, last, current):
            return {'h': state['h'] + 1}, current[:, None].expand(-1, 34, -1)
    class Readout:
        def exemplar(self, feats, mask):
            return torch.zeros(len(feats), 64, 32), torch.ones(len(feats), 64, dtype=torch.bool)
        def logits(self, ex, feats):
            result = torch.full((len(feats), 722), -10.)
            result[:, 10] = 10.
            return result
    class Quality:
        def __call__(self, x):
            assert x.shape[-1] == 12
            return torch.ones(len(x)) * 3
    loc.bb, loc.ro, loc.quality = Backbone(), Readout(), Quality()
    loc.bank = SimpleNamespace(eye_size=100,
        _render=lambda frame, crops: torch.ones(len(crops), 1, 721) * .123456)
    return loc


def track(tid, state):
    return SimpleNamespace(track_id=tid, state=state,
                           mean=np.asarray([200, 200, 20, 30, 1, 2, 3, 4.], dtype=np.float64))


def add_history(loc, tid):
    loc.hist[tid] = deque([(torch.ones(721) * .123456,
        np.asarray([195, 195, 205, 205.]), (150, 150, 250, 250))] * 4, maxlen=4)


def test_no_intervention_preserves_tracker_state_and_memory_is_continuous():
    loc = localizer(None)
    lost, tracked = track(1, 2), track(2, 1)
    add_history(loc, 1); add_history(loc, 2)
    before = lost.mean.copy(); tracked_before = tracked.mean.copy()
    loc.before_associate([lost, tracked], None, 1)
    assert np.array_equal(lost.mean, before) and np.array_equal(tracked.mean, tracked_before)
    assert set(loc.eyes) == {1} and torch.equal(loc.eyes[1][0], torch.ones(1, 721) * 4)
    loc.before_associate([lost, tracked], None, 1)
    assert torch.equal(loc.eyes[1][0], torch.ones(1, 721) * 5)
    assert loc.stats['births'] == 1 and loc.stats['refusals'] == 2
    loc.before_associate([], None, 1)
    assert not loc.eyes


def test_intervention_changes_only_lost_center_and_chunks_keep_track_identity():
    loc = localizer(.5)
    tracks = [track(i, 2) for i in range(1, 8)] + [track(8, 1)]
    for t in tracks:
        add_history(loc, t.track_id)
    before = {t.track_id: t.mean.copy() for t in tracks}
    loc.before_associate(tracks, None, 1)
    assert loc.stats['interventions'] == 7 and loc.stats['births'] == 7
    for t in tracks[:-1]:
        assert not np.array_equal(t.mean[:2], before[t.track_id][:2])
        assert np.array_equal(t.mean[2:], before[t.track_id][2:])
        assert torch.equal(loc.eyes[t.track_id][0], torch.ones(1, 721) * 4)
    assert np.array_equal(tracks[-1].mean, before[8])
