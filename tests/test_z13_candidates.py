import numpy as np
import pytest
from tracking.z13_candidates import candidates
from scripts.z13_target_model import center_and_presence


class Geometry:
    points=np.array([[i,0.] for i in range(9)])
    neighbors=np.array([[max(i-1,0),i,min(i+1,8)] for i in range(9)])
    spacing=1.


def test_first_candidate_exactly_preserves_center_and_score():
    g=Geometry();s=np.array([0,1,3,1,0,1,2,1,0.]);f=np.arange(27).reshape(9,3)
    r=candidates(s,f,g);c,p=center_and_presence(s,g)
    np.testing.assert_allclose(r['centers'][0],c,rtol=0,atol=1e-14)
    assert r['scores'][0]==p
    assert r['indices'].tolist()==[2,6]


def test_separation_and_ties_are_deterministic():
    r=candidates(np.ones(9),np.eye(9),Geometry())
    assert r['indices'].tolist()==[0,2,4]
    assert np.isfinite(r['features']).all()


def test_reject_nonfinite_features():
    with pytest.raises(ValueError):candidates(np.ones(9),np.full((9,2),np.nan),Geometry())
