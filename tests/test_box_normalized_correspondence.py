import numpy as np
from tracking.box_normalized_correspondence import box_transform


def test_larger_and_translated_box_preserves_relative_positions():
    old=np.array([5,10,15,30]);new=np.array([100,50,130,90]);t=box_transform(old,new)
    assert t['valid']
    points=np.array([new[:2],(new[:2]+new[2:])/2,new[2:]])
    recovered=old[:2]+(points-new[:2])/t['scale']
    assert np.allclose(recovered,[old[:2],(old[:2]+old[2:])/2,old[2:]])


def test_invalid_or_degenerate_boxes_abstain():
    assert not box_transform([0,0,0,10],[0,0,10,10])['valid']
    assert not box_transform([0,0,10,10],[0,0,np.nan,10])['valid']
