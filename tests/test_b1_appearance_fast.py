import numpy as np

from tracking.b1_appearance import relative_features as original
from tracking.b1_appearance_fast import relative_features


def test_exact_comparison_with_ties_missing_and_single_candidate():
    rng = np.random.default_rng(26)
    for width in (1, 2, 5, 18):
        for count in (0, 1, width):
            mask = np.arange(width) < count
            values = rng.integers(-2, 3, size=(width, 22)).astype(np.float32)
            values.ravel()[::7] = np.nan
            with np.errstate(invalid='ignore'):
                expected = original(values, mask)
            assert np.array_equal(relative_features(values, mask), expected, equal_nan=True)
