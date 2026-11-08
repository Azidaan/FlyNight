import numpy as np
import pytest
from scipy.spatial import Delaunay
from tracking.hex_background_filter import BackgroundPatchFilter,unit_patch,background_offsets


def test_target_only_is_exact_centered_similarity():
    rng=np.random.default_rng(12);a=rng.normal(size=(11,8));b=rng.normal(size=(11,8))
    model=BackgroundPatchFilter();assert model.fit(a,[])
    assert abs(model.score(b)-unit_patch(a)@unit_patch(b))<1e-12
    assert abs(model.score(a)-1)<1e-12


def test_dual_solution_matches_independent_primal_ridge():
    rng=np.random.default_rng(32);a=rng.normal(size=(5,2));negative=rng.normal(size=(7,5,2))
    model=BackgroundPatchFilter();assert model.fit(a,negative)
    X=np.stack([unit_patch(a)]+[unit_patch(p)/np.sqrt(7) for p in negative]);y=np.r_[1.,np.zeros(7)]
    w=np.linalg.solve(X.T@X+np.eye(10),X.T@y)
    assert np.allclose(w,model.weights,rtol=0,atol=1e-12)
    assert model.normalizer>=1/3-1e-12


def test_background_geometry_never_wraps_or_overlaps_target():
    x,y=np.meshgrid(np.arange(11.),np.arange(11.));points=np.column_stack([x.ravel(),y.ravel()])
    mesh=Delaunay(points);box=np.array([4.,4.,6.,6.]);idx=np.flatnonzero(((points>=box[:2])&(points<=box[2:])).all(1))
    offsets=background_offsets(points,idx,box,mesh,limit=8)
    assert len(offsets)==8
    for offset in offsets:
        shifted=box+np.tile(offset,2)
        assert np.maximum(0,np.minimum(shifted[2:],box[2:])-np.maximum(shifted[:2],box[:2])).prod()==0
        assert (mesh.find_simplex(points[idx]+offset)>=0).all()


def test_flat_target_abstains_and_flat_background_is_ignored():
    model=BackgroundPatchFilter();assert not model.fit(np.ones((5,8)),[])
    a=np.arange(40.).reshape(5,8);assert model.fit(a,[np.ones((5,8))])
    assert model.negative_count==0 and model.score(np.ones((5,8))) is None
    assert not model.fit(np.ones((5,8)),[])
    with pytest.raises(RuntimeError):model.score(a)
    with pytest.raises(ValueError):model.fit(a,[a.reshape(8,5)])
