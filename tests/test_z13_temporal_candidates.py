import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from tracking.z13_temporal_candidates import make_temporal, previous_inputs


def inputs(torch):
    torch.manual_seed(17)
    x = torch.randn(2, 3, 68); s = torch.randn(2, 68); g = torch.randn(2, 3, 8)
    m = torch.tensor([[True, True, False], [True, True, True]])
    return x, s, g, m, x.flip(1), g.flip(1), m.flip(1)


def test_zero_residual_exactly_replays_base():
    import torch
    model = make_temporal(torch); args = inputs(torch)
    torch.testing.assert_close(model(*args), model.base(*args[:4]), rtol=0, atol=0)


def test_candidate_and_memory_permutations_padding_and_gradient():
    import torch
    model = make_temporal(torch); args = inputs(torch)
    torch.nn.init.normal_(model.delta[-1].weight)
    x,s,g,m,px,pg,pm = args; perm = [1,0,2]
    score = model(*args)
    torch.testing.assert_close(model(x[:,perm],s,g[:,perm],m[:,perm],px,pg,pm), score[:,perm])
    torch.testing.assert_close(model(x,s,g,m,px[:,perm],pg[:,perm],pm[:,perm]), score)
    px=px.clone().requires_grad_(); model(x,s,g,m,px,pg,pm)[m].sum().backward()
    assert px.grad[pm].abs().sum()>0 and px.grad[~pm].abs().sum()==0
    altered=px.detach().clone(); altered[~pm]=9999
    torch.testing.assert_close(model(x,s,g,m,altered,pg,pm), score)


def test_geometry_cannot_read_any_activity():
    import torch
    model=make_temporal(torch,False); torch.nn.init.normal_(model.delta[-1].weight)
    x,s,g,m,px,pg,pm=inputs(torch)
    torch.testing.assert_close(model(x,s,g,m,px,pg,pm),model(x+999,s-999,g,m,px*999,pg,pm),rtol=0,atol=0)


def test_future_frames_cannot_change_previous_features():
    def frame(t):
        return dict(t=t,centers=np.array([[t,t],[t+1,t+1]]),features=np.full((2,68),t),
                    scores=np.array([2.,1.]),width=64.,seed_center=[0,0],seed_features=np.ones(68))
    seq={0:[frame(t) for t in range(3,8)]}; queries=[(0,1,[2,3])]
    before=previous_inputs(queries,seq)
    seq[0][-1]['features'][:]=999; seq[0][-2]['centers'][:]=999
    after=previous_inputs(queries,seq)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
