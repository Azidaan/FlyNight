import numpy as np
import pytest
from tracking.z13_observed_memory import observation_slots,memory_scores
from tracking.z13_target_center import observed_center_target,ridge_scores


def fixture():
    rng=np.random.default_rng(7)
    points=np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.],[1.,1.],[2.,1.]])
    feats=rng.normal(size=(4,6,3));search=rng.normal(size=(2,6,3))
    masks=np.zeros((4,6),bool);masks[:,[1,4]]=True
    return feats,masks,points,search


def test_padding_keeps_last_unique_observation():
    assert observation_slots([4,4,4,5])==[2,3]
    assert observation_slots([4,4,4,4])==[3]
    assert observation_slots([4,5,6,7])==[1,2,3]


def test_single_unique_frame_exactly_reproduces_prior_solver():
    f,m,p,q=fixture()
    actual=memory_scores(f,m,p,1.,q,[.01,.1],[4,4,4,4])
    expected=ridge_scores(f[3],observed_center_target(m[3],p,1.),q,[.01,.1])
    np.testing.assert_array_equal(actual,expected)


def test_separate_observed_masks_used_at_correct_times():
    f,m,p,q=fixture();m[1]=[True,False,False,True,False,False]
    actual=memory_scores(f,m,p,1.,q,[.1],[0,1,2,3])
    expected=ridge_scores(np.concatenate(f[1:]),np.concatenate([observed_center_target(v,p,1.) for v in m[1:]]),q,[.1])
    np.testing.assert_array_equal(actual,expected)
    wrong=m.copy();wrong[:]=m[3]
    assert not np.allclose(actual,memory_scores(f,wrong,p,1.,q,[.1],[0,1,2,3]))


def test_initialization_is_not_an_extra_memory_observation():
    f,m,p,q=fixture();a=memory_scores(f,m,p,1.,q,[.1],[0,1,2,3])
    f[0]=1e9;m[0]=False
    np.testing.assert_array_equal(a,memory_scores(f,m,p,1.,q,[.1],[0,1,2,3]))


def test_readout_api_does_not_accept_query_annotations():
    f,m,p,q=fixture()
    with pytest.raises(TypeError):memory_scores(f,m,p,1.,q,[.1],[0,1,2,3],query_gt=[0,0])
    with pytest.raises(ValueError):observation_slots([2,1,3,4])
