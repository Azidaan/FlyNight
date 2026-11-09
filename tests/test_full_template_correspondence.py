import numpy as np
import pytest
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import intersection_box,relative_template_plan


def offsets19():
    return np.array([(u+v/2,np.sqrt(3)*v/2) for u in range(-2,3)
        for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])


def test_exact_rectangle_intersection_matches_all_translations():
    box=np.array([10.,20,110,70]);off=offsets19()*5
    rng=np.random.default_rng(20261109);x=rng.uniform([0,0],[130,100],size=(10000,2))
    trial=x[:,None]+off
    brute=((trial>=box[:2])&(trial<=box[2:])).all((1,2))
    eroded=intersection_box(box,off)
    direct=((x>=eroded[:2])&(x<=eroded[2:])).all(1)
    np.testing.assert_array_equal(brute,direct)


def test_empty_intersection_is_not_replaced_with_nearest_cell():
    box=[0.,0,1,1];eroded=intersection_box(box,offsets19())
    assert (eroded[2:]<eroded[:2]).any()


def test_whole_support_preserves_old_cells_and_every_shift():
    points=np.array([(x,y) for x in range(21) for y in range(21)],float)
    reader=ObservedSpatialReader(points,20,[20,20]);args=([5,5,15,15],[0,0,20,20],[6,6,14,14],[0,0,20,20],offsets19())
    old=relative_template_plan(reader,*args,clip_to_candidate=True)
    full=relative_template_plan(reader,*args)
    assert len(full['cells'])==121 and len(old['cells'])<121
    assert set(old['cells'])<=set(full['cells'])
    positions=np.searchsorted(full['cells'],old['cells'])
    np.testing.assert_array_equal(old['query'],full['query'][positions])
    assert full['query'].shape==(121,19,2)
    assert (reader.mesh.find_simplex(full['query'])>=0).all()


def test_eye_boundary_still_limits_full_template():
    points=np.array([(x,y) for x in range(21) for y in range(21)],float)
    reader=ObservedSpatialReader(points,20,[20,20])
    full=relative_template_plan(reader,[0,0,10,10],[0,0,20,20],[0,0,10,10],[0,0,20,20],offsets19())
    assert full['eye_supported_cells']<full['reference_cells']
    assert (reader.mesh.find_simplex(full['query'])>=0).all()


def test_zero_shift_gives_same_support_in_both_modes():
    points=np.array([(x,y) for x in range(21) for y in range(21)],float)
    reader=ObservedSpatialReader(points,20,[20,20]);args=([5,5,15,15],[0,0,20,20],[6,6,14,14],[0,0,20,20],np.zeros((1,2)))
    old=relative_template_plan(reader,*args,clip_to_candidate=True)
    full=relative_template_plan(reader,*args)
    np.testing.assert_array_equal(old['cells'],full['cells'])
    np.testing.assert_array_equal(old['query'],full['query'])


def test_invalid_candidate_does_not_produce_a_plan():
    with pytest.raises(ValueError):intersection_box([0,0,-1,2],offsets19())
