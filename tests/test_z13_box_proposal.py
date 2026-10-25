from types import SimpleNamespace
import numpy as np
from tracking.z13_box_proposal import choose_region,proposal_costs


def test_constant_map_abstains_and_unsupported_peak_cannot_be_selected():
    points=np.array([[0.,0.],[1.,0.],[2.,0.]])
    boxes=np.array([[-.1,-.1,.1,.1],[1.9,-.1,2.1,.1]])
    selected,_,_=choose_region(np.ones(3),points,boxes,[True,True])
    assert selected==-1
    selected,_,valid=choose_region(np.array([3.,0.,9.]),points,boxes,[True,False])
    assert selected==-1 and not valid[1]


def test_peak_supported_region_is_chosen_from_the_observed_response_only():
    selected,p,_=choose_region(np.array([4.,0.,0.]),np.array([[0.,0.],[1.,0.],[2.,0.]]),[[-.1,-.1,.1,.1],[1.9,-.1,2.1,.1]],[True,True])
    assert selected==0 and p[0]>p[1]


def test_same_chosen_region_center_and_box_paths_change_only_specified_state():
    track=SimpleNamespace(mean=np.array([0.,0.,2.,4.,1.,2.,3.,4.]))
    detection=SimpleNamespace(xyxy=np.array([8.,7.,12.,13.]))
    before=track.mean.copy()
    def cost(tracks,_):return [tracks[0].mean[:4].copy()]
    a=proposal_costs(track,[detection],before[:4],cost,0,[0,.5,1],False)
    b=proposal_costs(track,[detection],before[:4],cost,0,[0,.5,1],True)
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[2],[10,10,2,4])
    np.testing.assert_array_equal(b[2],[10,10,4,6])
    np.testing.assert_array_equal(track.mean,before)
    assert all(np.array_equal(x,before[:4]) for x in proposal_costs(track,[detection],before[:4],cost,-1,[0,1],True))
