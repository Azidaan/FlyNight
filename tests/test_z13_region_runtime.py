import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from types import SimpleNamespace
import numpy as np
from tracking.z13_region_runtime import make_region_tracker,region_cost


def test_cost_adapter_only_runs_during_first_association_and_restores_on_error():
    class Base:
        fail=False
        def get_dists(self,tracks,detections):return np.full((1,2),.4)
        def _first_association(self,tracks,detections):
            answer=self.get_dists(tracks,detections)
            if self.fail:raise RuntimeError('Simulated downstream failure')
            return answer
    class Loc:
        def costs(self,tracks,detections,original):return original+.1
    tracker=make_region_tracker(Base,lambda:Loc())()
    np.testing.assert_allclose(tracker.get_dists([],[]),.4)
    np.testing.assert_allclose(tracker._first_association([],[]),.5)
    assert not tracker._regional_first
    tracker.fail=True
    import pytest
    with pytest.raises(RuntimeError):tracker._first_association([],[])
    assert not tracker._regional_first
    np.testing.assert_allclose(tracker.get_dists([],[]),.4)


def test_readout_costs_equal_frozen_hard_region_and_null_actions():
    import torch
    torch.set_default_device('cpu')
    class G:
        points=np.array([[4,4],[5,5],[6,6]],float);eye_size=10;spacing=2
        def nearest(self,p):return int(np.linalg.norm(self.points-p,axis=1).argmin())
    class Head:
        def __init__(self,values):self.values=values
        def __call__(self,*args):return torch.tensor([self.values],dtype=torch.float32)
    det=[SimpleNamespace(xyxy=[40,40,60,60],score=.8,cls=1)]
    original=np.array([.4],np.float32);args=(torch,Head([0,5]),np.ones((3,68)),np.ones(68),G(),[0,0,100,100],[50,50],det,1,original,0.)
    row,choice=region_cost(*args);np.testing.assert_allclose(row,[.2]);assert choice['detection']==0 and choice['action']=='region'
    alternate=list(args);alternate[1]=Head([5,0]);row,choice=region_cost(*alternate);np.testing.assert_array_equal(row,[1]);assert choice['action']=='unmatched'
    alternate[-1]=1.01;row,choice=region_cost(*alternate);np.testing.assert_array_equal(row,original);assert choice['action']=='unchanged'
    np.testing.assert_array_equal(original,np.array([.4],np.float32))


def test_receptor_runtime_preserves_history_and_original_readout():
    import copy
    import torch
    from tracking.z13_region_runtime import RegionalLocalizer
    from tracking.z13_ridge_localizer import CenterRidgeLocalizer
    torch.set_default_device('cpu')
    class Geometry:
        spacing=1.
        points=np.array([[0,0],[1,0],[0,1]],float)
        neighbors=np.array([[0,1,2],[1,0,2],[2,0,1]])
    runtime=RegionalLocalizer.__new__(RegionalLocalizer)
    runtime.torch=torch;runtime.arm='R1';runtime.geom=Geometry()
    runtime.mean=np.zeros(68);runtime.sd=np.linspace(.5,2,68)
    original=CenterRidgeLocalizer.__new__(CenterRidgeLocalizer)
    original.__dict__.update(runtime.__dict__)
    masks=np.array([[1,0,0],[0,1,1]],bool)
    runtime.lam=original.lam=.1
    observed=torch.arange(24,dtype=torch.float32).reshape(2,4,3)/24
    first=runtime.start_sequence(observed,masks);second=copy.deepcopy(first)
    for offset in (0.,.3):
        query=torch.tensor([[.1,.2,.9],[.7,.2,.1]],dtype=torch.float32)+offset
        a=runtime.advance(first,query);b=original.advance(second,query)
        for i,((point,score),(ref,confidence)) in enumerate(zip(a,b)):
            np.testing.assert_array_equal(point,ref);assert score==confidence
            np.testing.assert_array_equal(first[i]['state'],second[i]['state'])
            np.testing.assert_array_equal(first[i]['last'].numpy(),second[i]['last'].numpy())
            assert first[i]['query_features'].shape==(3,68)
