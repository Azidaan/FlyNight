import numpy as np
from tracking.z13_ridge_localizer import fit_ridge,ridge_response,receptor_step,R1_TAUS,augment
from tracking.z13_target_center import ridge_scores


def test_cached_observed_coefficients_equal_original_regression_for_many_queries():
    rng=np.random.default_rng(7);x=rng.normal(size=(721,68));y=rng.normal(size=721);queries=rng.normal(size=(8,721,68))
    for lam in (.001,.1):
        cached=ridge_response(fit_ridge(x,y,lam),queries)
        original=ridge_scores(x,y,queries,[lam])[0]
        assert np.array_equal(cached,original)


def test_incremental_receptor_state_equals_original_whole_history_features():
    from scripts.z13_target_model import receptor_features_rich,R1_TAUS as original_taus
    rng=np.random.default_rng(9);hx=rng.normal(size=(2,15,721));ema=np.repeat(hx[:,0,None,:],17,axis=1)
    reference=receptor_features_rich(hx);assert np.array_equal(R1_TAUS,original_taus)
    for t in range(15):
        ema,current=receptor_step(ema,hx[:,t]);assert np.array_equal(current,reference[t])
