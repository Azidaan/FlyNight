import numpy as np
from tracking.z13_region_choice import categorical_rows


def test_extreme_categorical_probabilities_reproduce_cost_interface():
    original=np.array([.2,.4,.6]);confidence=np.array([.9,.75,.3])
    rows,choices=categorical_rows(original,confidence,[False,True,True])
    assert choices==[-2,-1,1,2] and np.array_equal(rows[0],original)
    assert np.array_equal(rows[1],np.ones(3))
    np.testing.assert_allclose(rows[2],[1,.25,1]);np.testing.assert_allclose(rows[3],[1,1,.7])
    rows[0][0]=99;assert original[0]==.2


def test_no_visible_candidate_has_only_unchanged_and_null():
    rows,choices=categorical_rows([],[],[])
    assert choices==[-2,-1] and all(len(r)==0 for r in rows)
