import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pytest
import torch
from tracking.flyvis_state_transplant import transplant_nodes

def test_full_and_partial_state_are_isolated():
    a=torch.arange(12,dtype=torch.float64).reshape(2,6);b=-a-1
    x=transplant_nodes(a,b,[0,4]);assert torch.equal(x[:,[0,4]],b[:,[0,4]])
    assert torch.equal(x[:,[1,2,3,5]],a[:,[1,2,3,5]])
    assert torch.equal(transplant_nodes(a,b,list(range(6))),b)
    assert torch.equal(transplant_nodes(a,a,[0,4]),a)
    x.fill_(99);assert a[0,0]==0 and b[0,0]==-1

@pytest.mark.parametrize('ix,exc',[([0,0],ValueError),([],ValueError),([-1],IndexError),([6],IndexError)])
def test_invalid_selection_fails(ix,exc):
    with pytest.raises(exc):transplant_nodes(torch.ones(6),torch.zeros(6),ix)

def test_mismatched_or_invalid_donor_fails():
    with pytest.raises(ValueError):transplant_nodes(torch.ones(6),torch.zeros(5),[0])
    with pytest.raises(ValueError):transplant_nodes(torch.ones(6),torch.full((6,),float('nan')),[0])
