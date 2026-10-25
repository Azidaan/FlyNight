import numpy as np
from tracking.z13_ridge_uncertainty import ridge_posterior,region_probabilities
from tracking.z13_target_center import ridge_scores


def test_posterior_mean_is_identical_to_existing_ridge_and_variance_matches_direct_inverse():
    r=np.random.default_rng(7);x=r.normal(size=(17,4));y=r.random(17);s=r.normal(size=(2,9,4));lam=.3
    mean,sd,meta=ridge_posterior(x,y,s,lam)
    np.testing.assert_allclose(mean,ridge_scores(x,y,s,[lam])[0],rtol=0,atol=1e-14)
    c=s-x.mean(0);inv=np.linalg.inv((x-x.mean(0)).T@(x-x.mean(0))+len(x)*lam*np.eye(4))
    variance=meta['noise_variance']*np.einsum('...i,ij,...j->...',c,inv,c)
    np.testing.assert_allclose(sd**2,variance,rtol=1e-12,atol=1e-14)


def test_uncertainty_increases_in_unobserved_feature_direction():
    x=np.c_[np.linspace(-1,1,20),np.zeros(20)];y=np.sin(3*x[:,0]);s=np.array([[0.,0.],[0.,5.]])
    mean,sd,_=ridge_posterior(x,y,s,.1)
    assert mean[0]==mean[1] and sd[1]>sd[0]


def test_old_path_exact_and_constant_control_only_penalizes_shared_confidence():
    from tracking.z13_region_readout import region_probabilities as old
    p=np.array([[0,0],[1,0],[2,0],[3,0]],float);b=np.array([[0,0,1,0],[2,0,3,0],[9,0,10,0]],float)
    z=np.array([1.,2.,3.,4.]);v=np.ones(3,bool);u=np.array([0.,0.,1.,1.])
    original,valid=old(z,p,b,v);a,av=region_probabilities(z,p,b,v,u,'old')
    np.testing.assert_array_equal(original,a);np.testing.assert_array_equal(valid,av)
    zero,_=region_probabilities(z,p,b,v,np.zeros(4),'posterior');np.testing.assert_array_equal(zero,original)
    constant,_=region_probabilities(z,p,b,v,u,'constant');post,_=region_probabilities(z,p,b,v,u,'posterior')
    assert constant.sum()<original.sum() and post[1]/post[0]<constant[1]/constant[0] and post[2]==0
