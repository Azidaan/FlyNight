import numpy as np
from ultralytics.trackers.utils import matching
from tracking.free_singleton_assignment import free_singleton_matches


def solve(cost,plan,arm='U_GEO',alpha=.2,beta=0,states=None,tc=None,dc=None):
    cost=np.array(cost,np.float32);native,rr,cc=matching.linear_assignment(cost,thresh=.8)
    result=free_singleton_matches(cost,native,rr,cc,plan,alpha,beta,arm,
        states or [2]*len(cost),tc or [0]*len(cost),dc or [0]*cost.shape[1])
    return np.asarray(native).reshape(-1,2),result


def test_preserves_native_owner_when_bonus_would_steal():
    native,r=solve([[.3,1.],[.31,1.]],dict(votes=[[1,0]],columns={}),alpha=.4)
    assert r[0].tolist()==native.tolist() and r[4]==[]


def test_recovers_only_free_lost_row_and_keeps_geometry_state():
    native,r=solve([[.1,.9],[.9,.91]],dict(votes=[[1,1]],columns={}),states=[1,2])
    assert native.tolist()==[[0,0]] and r[0].tolist()==[[0,0],[1,1]]
    assert r[1]==[] and r[2]==[] and r[4]==[(1,1)]


def test_rejects_zero_overlap_cross_class_and_active_rescue():
    for cost,states,tc,dc in [([[1.]],[2],[0],[0]),([[.9]],[2],[0],[1]),([[.9]],[1],[0],[0])]:
        _,r=solve(cost,dict(votes=[[0,0]],columns={}),states=states,tc=tc,dc=dc)
        assert not len(r[0]) and not r[4]


def test_uniform_threshold_control_has_no_geometry_requirement():
    native,r=solve([[.1,.9],[.9,.91]],dict(votes=[],columns={}),arm='U_THRESHOLD',states=[1,2])
    assert native.tolist()==[[0,0]] and r[0].tolist()==[[0,0],[1,1]]


def test_zero_and_undefined_visual_scores_preserve_geometry_contract():
    plan=dict(votes=[[1,1]],columns={'1':dict(rows=[0,1],geometry_valid=True,scores={'G_FLYVIS':[None,.5]})})
    native,zero=solve([[.1,.9],[.9,.91]],plan,alpha=0,beta=0)
    assert zero[0].tolist()==native.tolist()
    _,plain=solve([[.1,.9],[.9,.91]],plan)
    _,visual=solve([[.1,.9],[.9,.91]],plan,arm='U_FLYVIS',beta=.4)
    np.testing.assert_array_equal(plain[0],visual[0]);np.testing.assert_array_equal(plain[3],visual[3])
