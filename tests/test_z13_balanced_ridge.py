import numpy as np
from tracking.z13_balanced_ridge import fit_observed,spatial_weights
from tracking.z13_ridge_localizer import fit_ridge,ridge_response


def test_disabled_weights_reproduce_original_ridge():
    rng=np.random.default_rng(8);x=rng.normal(size=(721,68));y=rng.uniform(size=721);mask=np.arange(721)<80;search=rng.normal(size=(3,721,68))
    assert np.allclose(ridge_response(fit_observed(x,y,mask,.1,False),search),ridge_response(fit_ridge(x,y,.1),search),atol=1e-12,rtol=1e-12)


def test_balanced_solution_satisfies_weighted_objective_stationarity():
    rng=np.random.default_rng(9);x=rng.normal(size=(100,8));y=rng.random(100);mask=np.arange(100)<17;lam=.3
    mu,w,b=fit_observed(x,y,mask,lam);q=spatial_weights(mask);error=(x-mu)@w+b-y
    assert np.isclose(q[mask].sum(),.5) and np.isclose(q[~mask].sum(),.5)
    assert abs(q@error)<1e-12
    assert np.max(np.abs(x.T@(q*error)+lam*w))<1e-12


def test_full_observed_mask_uses_uniform_fallback():
    assert np.array_equal(spatial_weights(np.ones(5,dtype=bool)),np.full(5,.2))
