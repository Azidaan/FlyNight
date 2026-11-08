import numpy as np
import pytest
from tracking.clipping_aware_correspondence import visible_edge_translation


def test_clipped_center_underestimates_motion_but_visible_edge_is_exact():
    old=np.array([70.,20,90,40]);new=np.array([90.,20,100,40])
    measured=(new[:2]+new[2:]-old[:2]-old[2:])/2
    assert measured[0]==15
    r=visible_edge_translation(old,new,measured,[100,80]);assert np.array_equal(r['translation'],[20,0])
    both=visible_edge_translation([80,20,100,40],[90,20,100,40],[5,3],[100,80])
    assert np.array_equal(both['translation'],[10,3])


def test_mirrored_horizontal_and_vertical_clipping():
    assert np.array_equal(visible_edge_translation([10,20,30,40],[0,20,10,40],[-15,2],[100,80])['translation'],[-20,2])
    assert np.array_equal(visible_edge_translation([20,50,40,70],[20,70,40,80],[2,15],[100,80])['translation'],[2,20])
    assert np.array_equal(visible_edge_translation([20,10,40,30],[20,0,40,10],[2,-15],[100,80])['translation'],[2,-20])


def test_unclipped_or_conflicting_clipping_preserves_prediction_exactly():
    d=np.array([.3,-7.1])
    r=visible_edge_translation([20,20,40,40],[21,21,41,41],d,[100,80])
    assert not r['changed'] and np.array_equal(r['translation'],d)
    r=visible_edge_translation([0,20,20,40],[80,20,100,40],d,[100,80])
    assert not r['changed'] and np.array_equal(r['translation'],d)


def test_invalid_geometry_is_rejected_without_mutating_input():
    d=np.array([1.,2]);visible_edge_translation([80,20,100,40],[90,20,100,40],d,[100,80]);assert np.array_equal(d,[1,2])
    with pytest.raises(ValueError):visible_edge_translation([2,3,1,4],[1,1,2,2],d,[100,80])
