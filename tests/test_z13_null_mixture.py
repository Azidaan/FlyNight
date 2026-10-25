import numpy as np
from tracking.z13_null_mixture import cost_row


def test_full_null_is_exact_noop_in_new_contract_and_not_in_original():
    old=np.array([.123,.6,1.]);p=np.zeros(3);c=np.array([.9,.8,.7]);s=np.ones(3,bool)
    np.testing.assert_array_equal(cost_row(old,p,c,s,.25,'null_mix'),old)
    assert cost_row(old,p,c,s,.25,'old')[0]>old[0]


def test_mixture_equals_explicit_expected_cost_over_unchanged_and_candidate_actions():
    old=np.array([.1,.6,1.]);p=np.array([.2,.3,0]);c=np.array([.9,.8,.7]);s=np.array([True,True,False]);alpha=.25
    expected=(1-alpha)*old+alpha*(1-p.sum())*old
    for j in range(2):
        action=np.ones(3);action[j]=1-c[j];expected+=alpha*p[j]*action
    expected[~s]=old[~s]
    np.testing.assert_allclose(cost_row(old,p,c,s,alpha,'null_mix'),expected,rtol=0,atol=1e-15)


def test_no_null_reproduces_prior_formula_and_alpha0_preserves_every_element():
    old=np.array([.2,.7]);p=np.array([.3,.7]);c=np.array([.9,.8]);s=np.ones(2,bool)
    np.testing.assert_allclose(cost_row(old,p,c,s,.25,'null_mix'),cost_row(old,p,c,s,.25,'old'),rtol=0,atol=1e-15)
    np.testing.assert_array_equal(cost_row(old,p,c,s,0,'null_mix'),old)
