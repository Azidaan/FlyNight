import numpy as np
import pytest
from scipy.spatial import Delaunay
from tracking.hex_linear_interpolation import sample_linear
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.hex_background_filter import unit_patch


def test_constants_survive_arbitrary_barycentric_coordinates_exactly():
    rng=np.random.default_rng(20261109);points=rng.uniform(-2,2,(47,2));mesh=Delaunay(points);triangles=mesh.simplices[rng.integers(0,len(mesh.simplices),311)];weights=rng.dirichlet([.7,1.3,2.1],311);query=(points[triangles]*weights[:,:,None]).sum(1)
    constants=np.array([.1,.3,1.,10.,-3.7,0.,1e-12,1e12]);values=np.broadcast_to(constants,(47,8))
    assert np.array_equal(sample_linear(mesh,values,query),np.broadcast_to(constants,(311,8)))
    affine=points@np.array([[.3,-.7],[1.2,.4]])+np.array([.1,-.3]);expected=query@np.array([[.3,-.7],[1.2,.4]])+np.array([.1,-.3])
    assert np.allclose(sample_linear(mesh,affine,query),expected,rtol=0,atol=3e-15)
    assert np.allclose(sample_linear(mesh,affine,points),affine,rtol=0,atol=3e-15)


def test_reader_rejects_flat_current_with_nonflat_reference_and_global_centering():
    xx,yy=np.meshgrid(np.arange(10.,100,20),np.arange(10.,100,20));points=np.column_stack([xx.ravel(),yy.ravel()]);reader=ObservedSpatialReader(points,100,[100,100]);plan=reader.plan([10,10,90,90],[0,0,100,100],[10,10,90,90],[0,0,100,100],[1.37,-2.19],[[0,0],[.17,.31]],use_visible_edge=False)
    reference=np.column_stack([np.sin(xx.ravel()/17),np.cos(yy.ravel()/13)]);current=np.tile([.1,.3],(25,1))
    assert len(plan['cells'])>=3
    for normalization in ['channels','global']:
        assert reader.compare(reference,current,plan,normalization=normalization)['score'] is None
    assert unit_patch(np.tile([.1,.3],(7,1))) is None
    a=np.column_stack([np.linspace(-1,1,7),np.full(7,.3)]);b=a.copy();b[:,1]=3.7
    assert np.array_equal(unit_patch(a),unit_patch(b))


def test_sampler_rejects_invalid_or_outside_queries():
    mesh=Delaunay([[0,0],[1,0],[0,1]])
    with pytest.raises(ValueError):sample_linear(mesh,np.ones((3,2)),[[1,1]])
    with pytest.raises(ValueError):sample_linear(mesh,np.ones((2,2)),[[.1,.1]])
    with pytest.raises(ValueError):sample_linear(mesh,np.ones((3,2)),[[np.nan,.1]])
