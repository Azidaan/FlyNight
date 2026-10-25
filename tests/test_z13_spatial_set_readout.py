import numpy as np
import torch
from tracking.z13_spatial_set_readout import make_model,region_coordinates


def fixture():
    points=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    full=torch.zeros((2,4,68));full[:,:,0]=torch.tensor([[1.,1.,-1.,-1.],[-1.,-1.,1.,1.]])
    rm=torch.ones((2,1,4),dtype=torch.bool);valid=torch.ones((2,1),dtype=torch.bool)
    means=full.mean(1)[:,None];seed=torch.zeros((2,68));geo=torch.zeros((2,1,14))
    c,s=region_coordinates(points.numpy(),rm[0].numpy(),1.)
    return means,seed,geo,valid,full,rm,torch.tensor(c)[None].repeat(2,1,1),torch.tensor(s)[None].repeat(2,1,1),points


def test_layout_is_observable_while_means_are_identical():
    args=fixture();assert torch.equal(args[0][0],args[0][1])
    torch.manual_seed(7);spatial=make_model(torch);mean=make_model(torch,'mean');mean.load_state_dict(spatial.state_dict())
    assert sum(p.numel() for p in spatial.parameters())==sum(p.numel() for p in mean.parameters())
    with torch.no_grad():a=spatial(*args);b=mean(*args)
    # Independent BLAS rows and reordered sums can differ by float32 roundoff.
    assert torch.allclose(b[0],b[1],atol=1e-7,rtol=1e-6) and abs(float(a[0]-a[1]))>1e-6


def test_joint_permutation_and_geometry_only_control():
    args=list(fixture());torch.manual_seed(0);m=make_model(torch);blank=make_model(torch,activity=False);blank.load_state_dict(m.state_dict())
    with torch.no_grad():old=m(*args);bg=blank(*args)
    ix=torch.tensor([2,0,3,1]);args[4]=args[4][:,ix];args[5]=args[5][:,:,ix];args[8]=args[8][ix]
    with torch.no_grad():new=m(*args);bg2=blank(*args)
    assert torch.allclose(old,new,atol=1e-7,rtol=1e-6) and torch.allclose(bg,bg2,atol=1e-7,rtol=1e-6)
    assert torch.allclose(bg[0],bg[1],atol=1e-7,rtol=1e-6)


def test_gradient_and_unequal_support():
    args=list(fixture());args[5]=args[5].clone();args[5][1,0,2:]=False
    for i in range(2):
        c,s=region_coordinates(args[8].numpy(),args[5][i].numpy(),1.)
        args[6][i]=torch.tensor(c);args[7][i]=torch.tensor(s)
    m=make_model(torch);z=m(*args);z.square().mean().backward()
    assert torch.isfinite(z).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(float(p.grad.abs().sum()) for p in m.point.parameters())>0
