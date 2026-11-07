import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from tracking.b1_learning import ReceptorHistory
from tracking.b3_object_memory import ObjectMemoryBranch


def test_memory_zero_start_and_gradient_to_observed_input_adapter():
    model=ObjectMemoryBranch(columns=7).cpu();backend=ReceptorHistory('cpu')
    history=torch.rand(1,5,257);candidates=torch.rand(1,3,256);base=torch.tensor([[.2,.7,.4]],dtype=torch.float64)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    probability,_=model(history,candidates,base,backend)
    assert torch.equal(probability,base)
    (-probability.mean()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    probability,_=model(history,candidates,base,backend);(-probability.mean()).backward()
    assert torch.isfinite(model.input.weight.grad).all() and model.input.weight.grad.abs().sum()>0
