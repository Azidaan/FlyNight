import numpy as np
import pytest
from types import SimpleNamespace
from tracking.z13_observed_contrast import contrast_scores,background_indices,quota_decisions


def test_same_target_and_background_produces_null_margin():
    target=np.ones(68);query=np.stack([target,-target])
    positive,contrast=contrast_scores(query,target,[target])
    np.testing.assert_allclose(contrast,0,atol=1e-15)
    assert positive[0]>positive[1]


def test_observed_background_suppresses_its_own_descriptor():
    target=np.eye(68)[0];background=np.eye(68)[1]
    _,scores=contrast_scores([target,background],target,[background])
    np.testing.assert_allclose(scores,[1,-1])
    np.testing.assert_allclose(contrast_scores([target],target,[])[0],contrast_scores([target],target,[])[1])


def test_background_neighborhood_must_avoid_entire_observed_mask():
    geom=SimpleNamespace(points=np.array([[0,0],[1,0],[2,0]],float),
        neighbors=np.array([[0,1],[1,0],[2,2]]),nearest=lambda c:int(np.argmin(np.abs(np.arange(3)-c[0]))))
    assert background_indices([[1,0],[2,0]],[True,False,False],geom)==[1]


def test_fixed_quota_is_stable_and_null_is_not_forced_to_intervene():
    values=np.ones(200)
    assert np.flatnonzero(quota_decisions(values)).tolist()==[0,1]
    assert not quota_decisions(np.zeros(200)).any()
    with pytest.raises(ValueError):quota_decisions(values,.05)
