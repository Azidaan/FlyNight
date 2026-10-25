import numpy as np
from tracking.z13_region_assignment_context import assignment_context


def test_context_uses_other_tracks_and_alternatives_without_mutating_costs():
    matrix=np.array([[.2,.7,.8],[.1,.6,.9]],np.float32);before=matrix.copy()
    actual=assignment_context(matrix,0,[-1,1,2])
    np.testing.assert_allclose(actual,[[.2,1,.2,0],[.7,.6,.2,.1],[.8,.9,.2,-.1]],atol=1e-7)
    np.testing.assert_array_equal(matrix,before)
    changed=matrix.copy();changed[1,1]=.3
    assert assignment_context(changed,0,[1])[0,1]<actual[1,1]


def test_no_competitor_no_detection_have_explicit_finite_conventions():
    np.testing.assert_array_equal(assignment_context(np.empty((1,0)),0,[-1]),[[1,1,1,0]])
    np.testing.assert_allclose(assignment_context(np.array([[.2]]),0,[0]),[[.2,1,1,-.8]])


def test_original_assignment_flags_are_observable_and_distinguish_null():
    from tracking.z13_region_assignment_context import assignment_state_context
    matrix=np.array([[.2,.7],[.1,.6]],np.float32)
    real=assignment_state_context(matrix,0,[-1,0,1],np.array([1,0],np.int32))
    np.testing.assert_array_equal(real[:,4:],[[1,0],[1,0],[1,1]])
    missing=assignment_state_context(matrix,0,[-1,0,1],np.array([-1,0],np.int32))
    np.testing.assert_array_equal(missing[:,4:],[[0,1],[0,0],[0,0]])
