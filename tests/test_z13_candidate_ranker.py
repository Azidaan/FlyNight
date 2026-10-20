import numpy as np
from tracking.z13_candidate_ranker import candidate_inputs,make_ranker


def example(torch):
    a=candidate_inputs([[1,2],[3,4]],np.arange(136).reshape(2,68),[2.,1.],[0,0],np.arange(68),[2,3],10,2)
    return tuple(torch.as_tensor(v[None]) for v in a)


def test_permutation_equivariance_and_padding():
    import torch
    torch.set_default_device('cpu');torch.manual_seed(0);m=make_ranker(torch)
    x,s,g,k=example(torch);p=m(x,s,g,k);perm=[1,0,2]
    q=m(x[:,perm],s,g[:,perm],k[:,perm])
    torch.testing.assert_close(q,p[:,perm]);assert p[0,2]<-1000


def test_geometry_ablation_cannot_read_activity():
    import torch
    torch.manual_seed(1);m=make_ranker(torch,False);x,s,g,k=example(torch)
    torch.testing.assert_close(m(x,s,g,k),m(x+100,s-20,g,k),rtol=0,atol=0)


def test_real_activity_reaches_output_and_gradient():
    import torch
    torch.manual_seed(2);m=make_ranker(torch,True);x,s,g,k=example(torch);x.requires_grad_(True)
    m(x,s,g,k)[0,:2].sum().backward()
    assert x.grad[0,:2].abs().sum()>0
    assert x.grad[0,2].abs().sum()==0
