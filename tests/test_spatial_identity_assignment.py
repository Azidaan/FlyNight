import numpy as np
from tracking.spatial_identity_assignment import corrected_cost,centered_similarity,flow_similarity
from ultralytics.trackers.utils import matching


def test_zero_is_exact_and_does_not_require_features():
    c=np.array([[.6,.9],[.4,.8]],np.float32)
    for arm in ['G','G_SHIFT','G_IMAGE','G_R1','G_DIS','G_FLYVIS','G_PERM']:
        assert np.array_equal(corrected_cost(c,[(0,0)],0,0,arm,{}),c)


def test_singleton_vote_reaches_global_competition():
    c=np.array([[.64],[.43]],np.float64)
    a=corrected_cost(c,[(0,0)],.3,0,'G',{})
    matches,_,_=matching.linear_assignment(a,thresh=.8)
    assert np.asarray(matches).tolist()==[[0,0]]
    assert np.array_equal(c,np.array([[.64],[.43]]))


def test_center_across_competing_tracks_not_singleton_candidate():
    c=np.full((2,1),.6);columns={0:dict(rows=[0,1],scores={'G_FLYVIS':[.8,-.2]})}
    a=corrected_cost(c,[],0,.4,'G_FLYVIS',columns)
    assert np.allclose(a[:,0],[.4,.8]) and abs((a-c).sum())<1e-12


def test_missing_rival_is_neutral_for_whole_column():
    c=np.full((2,1),.6);col={0:dict(rows=[0,1],scores={'G_FLYVIS':[.8,None]})}
    assert np.array_equal(corrected_cost(c,[(0,0)],.1,.4,'G_FLYVIS',col),corrected_cost(c,[(0,0)],.1,0,'G',{}))


def test_constant_identity_scores_are_neutral():
    c=np.full((2,1),.6);col={0:dict(rows=[0,1],scores={'G_FLYVIS':[.8,.8]})}
    assert np.array_equal(corrected_cost(c,[],0,.4,'G_FLYVIS',col),c)


def test_uniform_shift_has_no_feature_dependency():
    c=np.full((2,1),.6);col={0:dict(rows=[0,1],geometry_valid=True)}
    assert np.allclose(corrected_cost(c,[],0,.2,'G_SHIFT',col),.4)


def test_spatial_constant_is_undefined_but_flow_keeps_bulk_motion():
    x=np.ones((4,2))
    assert centered_similarity(x,x) is None
    assert flow_similarity(x,x,[1,1])==1


def test_one_to_one_with_negative_corrected_costs():
    c=np.array([[-.1],[-.2]])
    matches,ut,ud=matching.linear_assignment(c,thresh=.8)
    assert np.asarray(matches).tolist()==[[1,0]] and list(ut)==[0] and len(ud)==0
