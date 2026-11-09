import numpy as np
import pytest
from scipy.spatial import Delaunay
from tracking.channel_identity_head import (
    spatial_channel_features, causal_geometry, geometry_transform,
    objective, fit_ridge_logistic, decision_function,
)
from tracking.channel_balanced_correspondence import channel_balanced_similarity


def test_exact_gradient_and_hessian():
    rng = np.random.default_rng(20261109)
    x = rng.normal(size=(13,5)); y = np.arange(13) % 2
    theta = rng.normal(size=6); sw = np.linspace(.5,2,13)
    loss, gradient, hessian = objective(theta,x,y,sw)
    delta = 1e-5
    numeric_gradient = []
    numeric_hessian = []
    for k in range(6):
        step = np.eye(6)[k]*delta
        before = objective(theta-step,x,y,sw)
        after = objective(theta+step,x,y,sw)
        numeric_gradient.append((after[0]-before[0])/(2*delta))
        numeric_hessian.append((after[1]-before[1])/(2*delta))
    np.testing.assert_allclose(gradient,numeric_gradient,atol=1e-9)
    np.testing.assert_allclose(hessian,np.array(numeric_hessian).T,atol=1e-9)
    assert np.linalg.eigvalsh(hessian).min() > 0 and np.isfinite(loss)


def test_separable_samples_have_finite_regularized_solution():
    x = np.array([[-2.,0],[-1,0],[1,0],[2,0]])
    y = np.array([0,0,1,1])
    fit = fit_ridge_logistic(x,y)
    assert np.isfinite(fit['theta']).all() and fit['gradient_max'] < 1e-7
    np.testing.assert_array_equal(decision_function(fit,x)>0,y.astype(bool))
    assert fit['threshold'] == 0 and fit['regularization'] == 1


def test_missing_class_is_not_silently_fitted():
    with pytest.raises(ValueError,match='Both classes'):
        fit_ridge_logistic(np.ones((4,2)),np.ones(4))


def test_geometry_transform_excludes_test_values():
    train = np.array([[1.,3],[3,3]])
    test = np.array([[100.,-10]])
    x,y,stats = geometry_transform(train,test)
    np.testing.assert_array_equal(stats['mean'],[2.,3])
    np.testing.assert_array_equal(stats['scale'],[1.,1])
    np.testing.assert_array_equal(x,[[-1,0],[1,0]])
    np.testing.assert_array_equal(y,[[98,-13]])


def test_channel_vector_recovers_existing_score_and_shift():
    points = np.array([[0.,0],[1,0],[0,1],[1,1]])
    mesh = Delaunay(points)
    reference = np.array([[0.,1,7],[2,0,7],[1,3,7],[3,2,7]])
    current = reference.copy()
    cells = np.arange(4)
    query = np.stack([points,points[[1,0,3,2]]],axis=1)
    result = spatial_channel_features(mesh,reference,current,cells,query)
    expected = [channel_balanced_similarity(reference,current),
                channel_balanced_similarity(reference,current[[1,0,3,2]])]
    np.testing.assert_allclose(result['scores'],expected,atol=1e-14)
    assert result['winner'] == int(np.argmax(expected))
    assert result['channels'].mean() == pytest.approx(result['scalar'])
    assert result['channels'][-1] == 0


def test_constant_current_provides_no_identity_evidence():
    points = np.array([[0.,0],[1,0],[0,1],[1,1]])
    mesh = Delaunay(points)
    a = np.arange(12.).reshape(4,3)
    result = spatial_channel_features(mesh,a,np.ones_like(a),np.arange(4),points[:,None,:])
    assert not result['valid'] and result['channels'] is None and result['scalar'] is None


def test_geometry_has_no_absolute_position_or_uniform_scale_signal():
    a = np.array([10.,20,30,60]); b = np.array([15.,25,40,55])
    first = causal_geometry(a,b,.6,5,8)
    transformed = causal_geometry(a*3+100,b*3+100,.6,5,8)
    np.testing.assert_allclose(first,transformed)


def test_opposing_channels_can_reverse_wrong_equal_mean():
    x = np.array([[1.,-1.],[0.,.5]])
    assert x[0].mean() < x[1].mean()
    assert (x @ np.array([1.,0]))[0] > (x @ np.array([1.,0]))[1]
