"""Analytic transport and occlusion witnesses, independent of fitted identities."""
import numpy as np
import pytest
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.geometric_template_correspondence import geometric_template_plan
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear


def reader():
    y,x=np.mgrid[0:21,0:21]
    return ObservedSpatialReader(np.c_[x.ravel(),y.ravel()],20,[200,200])


def test_affine_signal_transport_matches_physical_object_coordinates():
    r=reader();rc=np.array([0.,0.,100.,100.]);ct=np.array([0.,0.,120.,100.])
    old=np.array([20.,20.,60.,60.]);box=np.array([35.,25.,75.,55.])
    scale=np.array([1.5,1.]);offset=np.array([5.,0.]);shifts=np.array([[0.,0.]])
    p=geometric_template_plan(r,old,rc,box,ct,shifts,scale,offset)
    reference=r.points*(rc[2:]-rc[:2])/r.eye_size+rc[:2]
    current_xy=r.points*(ct[2:]-ct[:2])/r.eye_size+ct[:2]
    # Manufactured current image stores exact canonical object coordinates.
    current=old[:2]+offset+(current_xy-box[:2])/scale
    observed=sample_linear(r.mesh,current,p['query'])[:,0,:]
    np.testing.assert_allclose(observed,reference[p['cells']],atol=3e-14,rtol=0)
    previous=relative_template_plan(r,old,rc,box,ct,shifts)
    previous_values=sample_linear(r.mesh,current,previous['query'])[:,0,:]
    assert abs(previous_values-reference[previous['cells']]).max()>5


def test_identical_geometry_preserves_ready_relative_plan():
    r=reader();old=[6,6,14,14];crop=[0,0,20,20];shifts=[[-2,0],[0,0],[2,0]]
    oldplan=relative_template_plan(r,old,crop,old,crop,shifts)
    new=geometric_template_plan(r,old,crop,old,crop,shifts,[1,1],[0,0])
    np.testing.assert_array_equal(new['cells'],oldplan['cells'])
    np.testing.assert_array_equal(new['query'],oldplan['query'])


def test_visibility_is_not_intersected_across_search_shifts():
    r=reader();old=[4,4,16,16];box=[8,4,16,16];crop=[0,0,20,20]
    p=geometric_template_plan(r,old,crop,box,crop,[[-4,0],[0,0],[4,0]],[1,1],[4,0])
    assert len(p['cells'])==9*13
    assert np.any(p['query'][:,:,0]>box[2])
    assert np.any(p['query'][:,:,0]<box[0])
    assert np.all(r.mesh.find_simplex(p['query'])>=0)


def test_eye_hull_is_required_for_every_shift_without_support_changes():
    r=reader();crop=[0,0,20,20]
    p=geometric_template_plan(r,[0,0,20,20],crop,[0,0,20,20],crop,[[0,0],[4,0]],[1,1],[0,0])
    assert len(p['cells'])==17*21
    assert p['query'].shape==(17*21,2,2)
    assert np.all(r.mesh.find_simplex(p['query'])>=0)


@pytest.mark.parametrize('scale,offset', [([0,1],[0,0]),([1,float('nan')],[0,0]),([1,1],[-1,0])])
def test_invalid_transform_cannot_silently_produce_evidence(scale,offset):
    r=reader()
    with pytest.raises(ValueError):
        geometric_template_plan(r,[1,1,10,10],[0,0,20,20],[1,1,10,10],[0,0,20,20],[[0,0]],scale,offset)
