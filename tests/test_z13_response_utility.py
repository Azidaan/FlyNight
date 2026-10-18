from types import SimpleNamespace
import numpy as np
from tracking.z13_response_utility import response_features


def geometry():
    points=np.array([(x,y) for x in range(5) for y in range(5)],float)
    # A fixed synthetic neighborhood is enough to test separated distractor ambiguity.
    neighbors=np.stack([np.array([i]*7) for i in range(25)])
    return SimpleNamespace(points=points,neighbors=neighbors)


def test_map_features_are_offset_invariant_and_geometry_translation_invariant():
    g=geometry();score=np.arange(25,dtype=float)
    expected=response_features(score,g)
    np.testing.assert_allclose(expected,response_features(score+77,g))
    shifted=SimpleNamespace(points=g.points+[123,44],neighbors=g.neighbors)
    np.testing.assert_allclose(expected,response_features(score,shifted))


def test_ambiguous_second_peak_and_flat_maps_are_finite():
    g=geometry();single=np.zeros(25);single[0]=5
    twin=single.copy();twin[-1]=5
    assert response_features(single,g)[2]>0
    assert response_features(twin,g)[2]==0
    flat=response_features(np.zeros(25),g)
    assert np.isfinite(flat).all() and abs(flat[3]-1)<1e-12
