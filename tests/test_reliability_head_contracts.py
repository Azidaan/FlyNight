"""Reliability head contracts independent of prior-untracked admission code."""
import numpy as np
import pytest

from tracking.reliability_head import BASE_FEATURES, FLY_FEATURES, LogisticHead, ReliabilityFunction

def spec(names=BASE_FEATURES):
    return dict(features=list(names), mean=np.zeros(len(names)).tolist(),
                scale=np.ones(len(names)).tolist(), coef=np.linspace(-.3, .5, len(names)).tolist(), intercept=.1)


@pytest.mark.parametrize('key,value', [
    ('scale', np.zeros(8)), ('scale', -np.ones(8)), ('scale', 1.),
    ('coef', np.full(8, np.nan)), ('coef', np.zeros(1)),
    ('mean', 0.), ('mean', np.full(8, np.inf)), ('intercept', float('nan')),
    ('features', ['box_w'] * 8), ('features', []),
])
def test_logistic_head_rejects_malformed_parameters(key, value):
    s = spec()
    s[key] = value
    with pytest.raises(ValueError):
        LogisticHead(s)


@pytest.mark.parametrize('part', ['base', 'fly'])
def test_reliability_rejects_wrong_feature_order(part):
    b, f = spec(), spec(BASE_FEATURES + FLY_FEATURES)
    s = b if part == 'base' else f
    s['features'][0], s['features'][1] = s['features'][1], s['features'][0]
    with pytest.raises(ValueError, match='feature'):
        ReliabilityFunction(LogisticHead(b), LogisticHead(f))


@pytest.mark.parametrize('x', [np.ones(8), np.ones((2, 1)), np.full((2, 8), np.nan)])
def test_logistic_head_rejects_malformed_feature_matrix(x):
    with pytest.raises(ValueError):
        LogisticHead(spec())(x)


def test_logistic_rejects_runtime_overflow_in_finite_parameters():
    s = spec()
    s['coef'] = np.full(8, 1e300).tolist()
    with pytest.raises(ValueError, match='nonfinite'):
        LogisticHead(s)(np.full((1, 8), 1e100))


def test_logistic_valid_math_exact_and_extreme_finite_scores():
    s = spec()
    model = LogisticHead(s)
    x = np.random.default_rng(31).normal(size=(32, 8))
    expected = 1 / (1 + np.exp(-((x - s['mean']) / s['scale'] @ s['coef'] + s['intercept'])))
    assert np.array_equal(expected, model(x))
    assert model(np.zeros((0, 8))).shape == (0,)
    s['coef'] = np.zeros(8).tolist(); s['intercept'] = -1000.
    assert LogisticHead(s)(x)[0] == 0.

