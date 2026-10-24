import numpy as np
import pytest
from types import SimpleNamespace
from scripts.z13_activation_timing_20261025 import correctness, shadow_center


def test_unknown_and_absent_identities_are_distinct():
    values = correctness([0, -1, -1, 0], ['a', 'b', 'c', 'a'], [True, True, False, True], ['a'])
    assert values.tolist() == [True, True, False, True]
    assert correctness([0], ['b'], [True], ['a']).tolist() == [False]


@pytest.mark.parametrize('fail', [False, True])
def test_shadow_center_restores_complete_state_even_on_error(fail):
    track = SimpleNamespace(mean=np.asarray([1.,2.,3.,4.,5.,6.,7.,8.]))
    saved = track.mean.copy()
    class Tracker:
        args = SimpleNamespace(match_thresh=.8)
        def get_dists(self, pool, detections):
            assert pool[0].mean[:2].tolist() == [9.,10.]
            assert np.array_equal(pool[0].mean[2:], saved[2:])
            if fail:
                raise RuntimeError('Synthetic cost failure')
            return np.asarray([[.1]])
    if fail:
        with pytest.raises(RuntimeError):
            shadow_center(Tracker(), [track], [], np.asarray([[.9]]), 0, [9.,10.])
    else:
        assert shadow_center(Tracker(), [track], [], np.asarray([[.9]]), 0, [9.,10.]).tolist() == [0]
    assert np.array_equal(track.mean, saved)
