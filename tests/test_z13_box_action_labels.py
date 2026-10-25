from types import SimpleNamespace
import numpy as np
import pytest
from tracking.z13_box_action_labels import full_box_rows


def test_complete_box_action_keeps_baseline_and_unmatched_distinct_and_restores_state():
    track=SimpleNamespace(mean=np.array([0.,0.,2.,4.,1.,2.,3.,4.]))
    det=SimpleNamespace(xyxy=np.array([8.,7.,12.,13.]))
    old=track.mean.copy()
    rows=full_box_rows(track,[det],np.array([.6]),lambda ts,ds:np.array([[ts[0].mean[:4].sum()]]),[-1,0])
    np.testing.assert_array_equal(rows[0],[.6])
    np.testing.assert_array_equal(rows[1],[1.])
    np.testing.assert_array_equal(rows[2],[30.])
    np.testing.assert_array_equal(track.mean,old)


def test_cost_failure_also_restores_state_and_invalid_choice_is_rejected():
    track=SimpleNamespace(mean=np.arange(8,dtype=float));old=track.mean.copy()
    def fail(*args):raise RuntimeError('cost error')
    with pytest.raises(RuntimeError):full_box_rows(track,[SimpleNamespace(xyxy=np.array([1,2,3,4]))],[.5],fail,[-1,0])
    np.testing.assert_array_equal(track.mean,old)
    with pytest.raises(ValueError):full_box_rows(track,[],[],fail,[0])
