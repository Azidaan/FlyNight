import numpy as np
import pytest
from tracking.z13_utility_transfer import fit_training_only


def test_disjoint_video_boundary_is_enforced():
    with pytest.raises(ValueError, match='overlap'):
        fit_training_only([[0], [1]], [0, 1], ['a', 'b'], ['b'])


def test_unknown_rows_cannot_change_training_normalization_or_predictions():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(21, 3))
    y = np.r_[np.tile([0, 1], 10), -1]
    videos = ['train'] * len(x)
    a, info = fit_training_only(x, y, videos, ['dev'])
    changed = x.copy()
    changed[-1] = 1e9
    b, _ = fit_training_only(changed, y, videos, ['dev'])
    np.testing.assert_array_equal(a.predict_proba(x[:20]), b.predict_proba(x[:20]))
    assert info['fit_known'] == 20 and info['excluded_unknown'] == 1


def test_saved_estimator_predicts_without_evaluation_outcomes(tmp_path):
    import joblib
    x = np.arange(12, dtype=float).reshape(6, 2)
    model, _ = fit_training_only(x, [0, 0, 1, 0, 1, 1], ['train'] * 6, ['dev'])
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict_proba(x), joblib.load(path).predict_proba(x))
