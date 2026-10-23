import torch
from tracking.z13_search_conditioned import make_predictor
torch.set_num_threads(2)


def inputs():
    torch.manual_seed(12)
    return torch.randn(4,3,68),torch.randn(4,68),torch.randn(4,3,8),torch.tensor([[1,1,1],[1,1,0],[1,0,0],[1,1,1]],dtype=torch.bool)


def test_weights_query_dependency_only_when_enabled():
    x,s,g,m=inputs();a=make_predictor(torch,True,True);b=make_predictor(torch,True,False);b.load_state_dict(a.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert torch.equal(b.parts(x,s,g,m)[0],b.parts(x+2,s,g+2,m)[0])
    assert not torch.allclose(a.parts(x,s,g,m)[0],a.parts(x+2,s,g+2,m)[0])


def test_permutation_padding_and_batch_independence():
    x,s,g,m=inputs();p=torch.tensor([2,0,1]);inv=torch.argsort(p)
    for condition in (False,True):
        model=make_predictor(torch,True,condition).eval();out=model(x,s,g,m)
        torch.testing.assert_close(out,model(x[:,p],s,g[:,p],m[:,p])[:,inv],atol=2e-6,rtol=2e-6)
        x2=x.clone();g2=g.clone();x2[~m]=100;g2[~m]=-100
        torch.testing.assert_close(out,model(x2,s,g2,m),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(out[:1],model(x[:1],s[:1],g[:1],m[:1]),atol=2e-6,rtol=2e-6)


def test_geometry_control_and_training_save_reload(tmp_path):
    x,s,g,m=inputs();model=make_predictor(torch,False,True)
    torch.testing.assert_close(model(x,s,g,m),model(x*19,s+35,g,m))
    opt=torch.optim.AdamW(model.parameters(),lr=.001);before=model(x,s,g,m).detach()
    loss=model(x,s,g,m)[m].square().mean();loss.backward();opt.step()
    assert not torch.equal(before,model(x,s,g,m))
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path);copy=make_predictor(torch,False,True);copy.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
    torch.testing.assert_close(model(x,s,g,m),copy(x,s,g,m))
