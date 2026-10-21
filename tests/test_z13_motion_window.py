import numpy as np
import pytest
from tracking.z13_motion_window import motion_window_scores


def test_zero_window_preserves_original_map_exactly():
    scores=np.array([-3.,5.,1.]);xy=np.array([[-1.,0.],[0.,0.],[1.,0.]])
    assert np.array_equal(motion_window_scores(scores,xy,[1.,0.],4.,0.),scores)


def test_window_prefers_forecast_and_is_invariant_to_positive_map_rescaling():
    scores=np.array([10.,2.,0.]);xy=np.array([[-1.,0.],[0.,0.],[1.,0.]])
    only_window=motion_window_scores(scores,xy,[1.,0.],4.,1.)
    assert np.argmax(only_window)==2 and np.isclose(only_window.sum(),1.)
    a=motion_window_scores(scores,xy,[1.,0.],4.,.25)
    b=motion_window_scores(3*scores+15,xy,[1.,0.],4.,.25)
    assert np.allclose(a,b)
    unsupported=motion_window_scores(scores,xy,[100.,100.],4.,1.)
    assert np.argmax(unsupported)==np.argmax(scores)


def test_flat_or_invalid_map_is_handled_explicitly():
    xy=np.array([[-1.,0.],[0.,0.],[1.,0.]])
    result=motion_window_scores(np.ones(3),xy,[0.,0.],4.,.5)
    assert np.isfinite(result).all() and np.isclose(result.sum(),1.)
    with pytest.raises(ValueError):motion_window_scores([1.,np.nan,0.],xy,[0.,0.],4.,.5)
