import numpy as np
import pytest
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tracking.z13_pairwise_utility import pair_indices, loss_gradient, leave_video_out


def data():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] + .4 * rng.normal(size=len(x)) > 0).astype(int)
    videos = np.repeat(['a', 'b', 'c'], 40)
    return x, y, videos


def test_gradient_for_both_fixed_objectives():
    x, y, _ = data()
    pos, neg = pair_indices(y, limit=100)
    diff = x[pos] - x[neg]
    theta = np.array([.2, -.1, .3, .4, -.5])
    for strength in (0., 1.):
        error = check_grad(lambda t: loss_gradient(t, x, y, diff, strength)[0],
                           lambda t: loss_gradient(t, x, y, diff, strength)[1], theta)
        assert error < 1e-6


def test_pairs_are_bounded_distinct_and_reproducible():
    y = np.r_[np.ones(100), np.zeros(100)]
    a, b = pair_indices(y)
    assert len(a) == 4096 and len(set(zip(a, b))) == 4096
    assert np.all(y[a] == 1) and np.all(y[b] == 0)
    c, d = pair_indices(y)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError):
        pair_indices(np.ones(10))


def test_point_objective_matches_high_precision_sklearn():
    x, y, videos = data()
    scores, records = leave_video_out(x, y, videos, strength=0.)
    for record in records:
        test = videos == record['test_video']
        scaler = StandardScaler().fit(x[~test])
        model = LogisticRegression(C=1, solver='lbfgs', max_iter=500, tol=1e-10)
        model.fit(scaler.transform(x[~test]), y[~test])
        np.testing.assert_allclose(scores[test], model.predict_proba(scaler.transform(x[test]))[:, 1], atol=2e-5)


def test_unknowns_do_not_enter_fit_or_scaler():
    x, y, videos = data()
    y[::10] = -1
    first, records = leave_video_out(x, y, videos, strength=1.)
    changed = x.copy()
    changed[y < 0] = 999
    second, _ = leave_video_out(changed, y, videos, strength=1.)
    np.testing.assert_array_equal(first[y >= 0], second[y >= 0])
    for r in records:
        fit = (videos != r['test_video']) & (y >= 0)
        np.testing.assert_allclose(r['mean'], x[fit].mean(0))
        assert r['test_video'] not in r['fit_videos']


def test_held_out_outcomes_cannot_change_own_predictions():
    x, y, videos = data()
    scores, _ = leave_video_out(x, y, videos, strength=1.)
    changed = y.copy()
    changed[videos == 'a'] = 1 - changed[videos == 'a']
    other, _ = leave_video_out(x, changed, videos, strength=1.)
    np.testing.assert_array_equal(scores[videos == 'a'], other[videos == 'a'])


def test_cancellation_is_checked_during_optimization():
    x, y, videos = data()
    calls = []
    def cancel():
        calls.append(True)
        if len(calls) > 1:
            raise TimeoutError('budget')
    with pytest.raises(TimeoutError):
        leave_video_out(x, y, videos, strength=1., check=cancel)
