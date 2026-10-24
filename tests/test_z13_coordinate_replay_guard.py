import numpy as np
import pytest
from scripts.z13_coordinate_replay_guard import compare_outputs


def example():
    return np.array([[0., 10., 20., 30., 40., 7., .75, 2.],
                     [1., 11., 21., 31., 41., 7., .75, 2.]])


def test_coordinate_only_guard_accepts_observed_roundoff_without_mutation():
    old = example()
    new = old.copy()
    new[0, 1] += 0.000030517578125
    before = new.copy()
    result = compare_outputs(new, old)
    assert result['accepted'] and not result['exact']
    assert result['coordinate_max_abs'] == 0.000030517578125
    assert np.array_equal(new, before)


@pytest.mark.parametrize('column', [0, 5, 6, 7])
def test_noncoordinate_fields_remain_exact(column):
    old = example()
    new = old.copy()
    new[0, column] += 1e-8
    with pytest.raises(AssertionError):
        compare_outputs(new, old)


def test_absolute_guard_has_no_relative_tolerance_and_rejects_changed_rows():
    old = example()
    old[:, 1:5] += 100000.
    new = old.copy()
    new[0, 1] += 0.00011
    with pytest.raises(AssertionError):
        compare_outputs(new, old)
    with pytest.raises(AssertionError):
        compare_outputs(old[:1], old)
    with pytest.raises(AssertionError):
        compare_outputs(old[::-1], old)
    new = old.copy()
    new[0, 1] = np.nan
    with pytest.raises(AssertionError):
        compare_outputs(new, old)
