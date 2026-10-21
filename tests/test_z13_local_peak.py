import numpy as np
from tracking.z13_local_peak import quadratic_peak
from scripts.z13_target_model import center_and_presence


class Geometry:
    spacing=1.
    points=np.vstack([[0,0],np.stack([np.cos(np.arange(6)*np.pi/3),np.sin(np.arange(6)*np.pi/3)],1)])
    neighbors=np.tile(np.arange(7),(7,1))


def test_recovers_off_grid_parabolic_vertex_with_unchanged_confidence():
    g=Geometry();truth=np.array([.2,.1]);score=3-np.square(g.points-truth).sum(1)
    center,p,kind=quadratic_peak(score,g);old,oldp=center_and_presence(score,g)
    np.testing.assert_allclose(center,truth,atol=1e-7);assert p==oldp and kind=='quadratic'
    assert np.linalg.norm(old-truth)>1e-3


def test_translation_and_scale_equivariance():
    g=Geometry();score=3-np.square(g.points-[.2,.1]).sum(1)
    a,_,_=quadratic_peak(score,g)
    class Transformed(Geometry):
        points=g.points*10+[100,200];spacing=10.
    b,_,_=quadratic_peak(score,Transformed())
    np.testing.assert_allclose(b,a*10+[100,200],atol=1e-6)


def test_flat_and_insufficient_support_fall_back_exactly():
    g=Geometry();score=np.zeros(7);old,p=center_and_presence(score,g)
    c,q,kind=quadratic_peak(score,g);np.testing.assert_array_equal(c,old);assert q==p and kind!='quadratic'
    g.neighbors=np.tile([0,1,1],(7,1));old,p=center_and_presence(score,g)
    c,q,kind=quadratic_peak(score,g);np.testing.assert_array_equal(c,old);assert q==p and kind=='insufficient_support'
