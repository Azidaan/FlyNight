import numpy as np
from tests.test_z13_boundary_readout import geometry
from tracking.z13_absolute_confidence import read_response
from tracking.z13_ridge_localizer import map_center


def test_raw_scale_is_retained_but_coordinates_are_unchanged():
    g=geometry();x=np.sin(np.arange(721));x[200]=9
    center,score,allowed=read_response(x,g)
    old,old_z=map_center(x,g)
    assert np.array_equal(center,old) and score==9 and allowed
    assert read_response(.01*x,g)[1]==.09
    assert read_response(x,g,False)[1]==old_z


def test_constant_map_has_finite_raw_confidence():
    g=geometry();center,score,allowed=read_response(np.full(721,.3),g)
    assert score==.3 and np.isfinite(center).all() and allowed
