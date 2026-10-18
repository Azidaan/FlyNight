import numpy as np
import torch
from tracking.z13_target_center import ridge_scores
from tracking.z13_target_center_torch import ridge_scores_torch
from tracking.z13_utility import observable_features, leave_video_out


def test_ridge_parity_and_gradient():
    rng = np.random.default_rng(4)
    x, y, search = rng.normal(size=(12, 4)), rng.random(12), rng.normal(size=(2, 12, 4))
    a = torch.tensor(x, requires_grad=True)
    b = torch.tensor(search, requires_grad=True)
    scores = ridge_scores_torch(a, torch.tensor(y), b, [.01, .1])
    np.testing.assert_allclose(scores.detach(), ridge_scores(x, y, search, [.01, .1]), atol=1e-12)
    scores.square().sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.norm() > 0
    assert torch.isfinite(b.grad).all() and b.grad.norm() > 0


def test_ground_truth_fields_cannot_enter_observable_features():
    row = dict(center=[3, 2], pred=[1, 1], pos=2, p_present=1.)
    a = observable_features(row, [0, 0, 10, 20])
    row.update(kind=1, useful=True, err=999, err_pred=1, scale=1e20, gtbox=[1]*4)
    b = observable_features(row, [0, 0, 10, 20])
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_held_out_video_outcomes_do_not_change_its_own_predictions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 3)); videos = np.repeat(['a', 'b', 'c'], 8)
    targets = np.tile([0, 1], 12)
    a, folds = leave_video_out(x, targets, videos)
    changed = targets.copy(); changed[videos == 'a'] = 1 - changed[videos == 'a']
    b, _ = leave_video_out(x, changed, videos)
    assert np.array_equal(a[videos == 'a'], b[videos == 'a'])
    assert all(f['test_video'] not in f['fit_videos'] for f in folds)
