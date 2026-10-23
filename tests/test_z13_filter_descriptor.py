import numpy as np
from tracking.z13_filter_descriptor import fit_descriptor,make_ranker
from tracking.z13_target_center import ridge_scores


def test_fit_matches_existing_and_background_matters():
    rng=np.random.default_rng(12);x=rng.normal(size=(45,68));q=rng.normal(size=(3,45,68));y=np.zeros(45);y[:7]=1
    w,mu,b=fit_descriptor(x,y,.1)
    np.testing.assert_allclose((q-mu)@w+b,ridge_scores(x,y,q,[.1])[0],atol=1e-12,rtol=0)
    other=x.copy();other[7:]+=rng.normal(size=(38,68));v,_,_=fit_descriptor(other,y,.1)
    np.testing.assert_array_equal(other[:7].mean(0),x[:7].mean(0));assert not np.allclose(w,v)


def test_ranker_masks_permutation_geometry_and_reload():
    import os;os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_default_device('cpu');torch.manual_seed(4)
    x=torch.randn(2,3,68);s=torch.randn(2,68);g=torch.randn(2,3,8);m=torch.tensor([[True,True,False],[True,True,True]])
    model=make_ranker(torch);a=model(x,s,g,m);order=[2,0,1]
    torch.testing.assert_close(model(x[:,order],s,g[:,order],m[:,order]),a[:,order])
    x2=x.clone();x2[0,2]=999;torch.testing.assert_close(model(x2,s,g,m),a)
    model2=make_ranker(torch);model2.load_state_dict(model.state_dict());torch.testing.assert_close(model2(x,s,g,m),a)
    ab=make_ranker(torch,False);torch.testing.assert_close(ab(x,s,g,m),ab(x*12,s*9,g,m))
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in ab.parameters())
