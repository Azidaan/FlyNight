import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
import pytest
from tracking.flyvis_cached_readouts import make_model, KINDS

@pytest.mark.parametrize('kind',KINDS)
def test_zero_residual_preserves_scores_and_candidate_permutation(kind):
    model=make_model(torch,kind,42)
    g=torch.arange(64,dtype=torch.float64).reshape(4,2,8)/100
    mu=torch.zeros(8,dtype=torch.float64);sd=torch.ones(8,dtype=torch.float64)
    before=g.clone();out=model(g,mu,sd)
    assert torch.equal(out,g.sum(-1)) and torch.equal(before,g)
    assert torch.equal(model(g.flip(1),mu,sd),out.flip(1))
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out[:,0]-out[:,1],torch.ones(4,dtype=torch.float64))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.device.type=='cpu' for p in model.parameters()) and not torch.cuda.is_initialized()

def test_sigma_pi_matches_paper_equation_for_signed_inputs():
    model=make_model(torch,'sigma_pi',42)
    with torch.no_grad():
        model.w.copy_(torch.arange(8,dtype=torch.float64)/10)
        model.b.fill_(.7);model.m.fill_(.4);model.gamma.fill_(.3)
        model.alpha.copy_(torch.linspace(-1,1,8,dtype=torch.float64))
    x=np.linspace(-2,2,8)[None]
    a=1/(1+np.exp(-model.alpha.detach().numpy()));mix=1/(1+np.exp(-.3))
    expected=x.sum(-1)+(1-mix)*(x@np.arange(8)/10+.7)+mix*.4*np.prod(1-a+a*x,axis=-1)
    actual=model(torch.tensor(x),torch.zeros(8,dtype=torch.float64),torch.ones(8,dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
