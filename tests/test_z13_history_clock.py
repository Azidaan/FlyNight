from types import SimpleNamespace
import numpy as np
import torch
from tracking.z13_history_clock import clock_features,frame_intervals


class Nodes(dict):
    def __getattr__(self,name):return self[name]


class Backbone:
    def __init__(self,n):self.index=torch.arange(n)[None];self.net=SimpleNamespace(_state_api=lambda s:s)
    def init(self,h):return SimpleNamespace(nodes=Nodes(activity=h[:,0].clone()),edges=Nodes(trace=h[:,0].clone()))
    def step(self,state,last,current):
        s=SimpleNamespace(nodes=Nodes(activity=.5*state.nodes.activity+.5*current),edges=Nodes(trace=.25*state.edges.trace+.75*last))
        return s,s.nodes.activity[:,self.index]


def test_variable_elapsed_time_matches_manual_individual_interpolation_and_zero_skip():
    x=torch.tensor([[[0.,1.],[0.,1.],[2.,3.],[4.,5.]],[[1.,0.],[2.,1.],[2.,1.],[5.,4.]]]);frames=np.array([[0,0,2,3],[0,1,1,4]])
    nb=np.array([[0,1],[1,0]]);result=clock_features(torch,'D',Backbone(2),x,nb,{1,2,3},frames)
    for b in range(2):
        state=x[b,0].clone()
        for t in range(1,4):
            n=frames[b,t]-frames[b,t-1]
            for k in range(1,n+1):state=.5*state+.5*(x[b,t-1]*(1-k/n)+x[b,t]*(k/n))
            np.testing.assert_array_equal(result[t][b,0],state.numpy())


def test_unit_frame_intervals_reproduce_original_backbone_sequence_exactly():
    from scripts.z13_target_model import episode_features
    x=torch.arange(24,dtype=torch.float32).reshape(2,4,3)/20;nb=np.array([[0,1],[1,2],[2,0]])
    a=episode_features(torch,'D',Backbone(3),x,nb,{1,2,3});b=clock_features(torch,'D',Backbone(3),x,nb,{1,2,3},np.tile(np.arange(4),(2,1)))
    for t in a:np.testing.assert_array_equal(a[t],b[t])


def test_rich_retina_unit_clock_is_exact_original_and_invalid_time_rejected():
    from scripts.z13_target_model import episode_features
    x=torch.arange(24,dtype=torch.float32).reshape(2,4,3)/20;nb=np.array([[0,1],[1,2],[2,0]])
    a=episode_features(torch,'R1',None,x,nb,{1,2,3});b=clock_features(torch,'R1',None,x,nb,{1,2,3},np.tile(np.arange(4),(2,1)))
    for t in a:np.testing.assert_array_equal(a[t],b[t])
    import pytest
    with pytest.raises(ValueError):frame_intervals(np.array([[0,2,1]]),(1,3))
