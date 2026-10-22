import numpy as np
from scipy.optimize import minimize
from tracking.z13_hinge_target import fit_hinge,objective
from tracking.z13_balanced_ridge import fit_observed,spatial_weights


def test_analytic_gradient_matches_independent_finite_difference():
    rng=np.random.default_rng(7);z=np.column_stack([rng.normal(size=(50,6)),np.ones(50)]);target=rng.random(50);mask=np.arange(50)<11;q=spatial_weights(mask);theta=rng.normal(size=7)
    value,gradient,_,_=objective(theta,z,target,mask,q,.2)
    def scalar(t):
        error=z@t-target
        return .5*np.sum(q*np.where(mask,error**2,np.maximum(error,0)**2))+.1*np.sum(t[:-1]**2)
    h=1e-6;numeric=np.array([(scalar(theta+np.eye(7)[i]*h)-scalar(theta-np.eye(7)[i]*h))/(2*h) for i in range(7)])
    assert np.isclose(value,scalar(theta)) and np.allclose(gradient,numeric,rtol=1e-6,atol=1e-8)


def test_solution_agrees_with_independent_optimizer_and_preserves_target():
    rng=np.random.default_rng(31);x=rng.normal(size=(721,12));mask=np.arange(721)<70;y=np.exp(-np.arange(721)/70);before=y.copy();lam=.1
    (mu,w,b),info=fit_hinge(x,y,mask,lam);q=spatial_weights(mask);z=np.column_stack([x-mu,np.ones(721)])
    def scalar(t):
        delta=z@t-y
        return .5*np.sum(q*np.where(mask,delta**2,np.maximum(delta,0)**2))+.5*lam*np.sum(t[:-1]**2)
    independent=minimize(scalar,np.zeros(13),method='BFGS',options={'gtol':1e-8,'maxiter':500})
    assert abs(scalar(np.r_[w,b])-independent.fun)<1e-9
    assert info['gradient_max']<=info['tolerance'] and info['monotone']
    assert np.array_equal(y,before)


def test_easy_negative_background_can_remain_negative():
    x=np.array([[1.],[0.],[-3.],[-5.]]);y=np.array([1.,0.,0.,0.]);mask=np.array([True,False,False,False])
    (mu,w,b),info=fit_hinge(x,y,mask,.01)
    square=fit_observed(x,y,mask,.01)
    assert ((x-mu)@w+b)[-1]<0
    assert info['final_objective']<info['initial_objective'] and np.linalg.norm(w-square[1])>1e-4
