import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from tracking.z13_activity_amplitude import make_amplitude_ranker


def test_only_query_normalization_changes_with_equal_capacity_and_initial_weights():
    torch.manual_seed(1);a=make_amplitude_ranker(torch,preserve=False)
    torch.manual_seed(1);b=make_amplitude_ranker(torch,preserve=True)
    assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items())
    x=torch.arange(68,dtype=torch.float64)
    norm=a.embed[0].double();affine=b.embed[0].double()
    assert torch.allclose(norm(x),norm(x+4),atol=1e-10,rtol=0)
    assert not torch.allclose(affine(x),affine(x+4))


def test_real_readout_forward_backward_stays_cpu_and_finite():
    m=make_amplitude_ranker(torch,preserve=True)
    x=torch.randn(3,4,68);seed=torch.randn(3,68);geo=torch.randn(3,4,14);mask=torch.ones(3,4,dtype=torch.bool)
    loss=m(x,seed,geo,mask).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert not torch.cuda.is_initialized()
