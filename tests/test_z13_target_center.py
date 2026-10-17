import numpy as np
import pytest
from scripts.z13_target_model import fit_predict
from tracking.z13_target_center import observed_center_target, ridge_scores


def test_binary_control_reproduces_original():
    rng=np.random.default_rng(7); x=rng.normal(size=(31,8)); xs=rng.normal(size=(3,31,8)); mask=np.arange(31)<7
    assert np.array_equal(ridge_scores(x,mask,xs,[.01,.1]),fit_predict(x,mask,xs,[.01,.1]))


def test_target_translation_and_center_peak():
    points=np.array([(x,y) for x in range(-2,3) for y in range(-2,3)],float)
    mask=(abs(points[:,0])<=1)&(abs(points[:,1])<=1)
    target=observed_center_target(mask,points,1)
    assert np.array_equal(target,observed_center_target(mask,points+[8,9],1))
    assert np.array_equal(points[target.argmax()],[0,0])
    assert target.max()==1 and target.min()>0
    assert target[0]<target[6]<target[12]


def test_single_column_finite_empty_rejected():
    p=np.array([[0,0],[1,0],[0,1]],float)
    assert np.isfinite(observed_center_target([True,False,False],p,1)).all()
    with pytest.raises(ValueError): observed_center_target([False]*3,p,1)


def test_ridge_can_learn_nonconstant_center_without_search_labels():
    points=np.array([(x,y) for x in range(-2,3) for y in range(-2,3)],float)
    y=observed_center_target(np.ones(25,bool),points,1)
    out=ridge_scores(np.eye(25),y,np.eye(25)[None],[.001])[0,0]
    assert out.argmax()==y.argmax() and np.std(out)>0
