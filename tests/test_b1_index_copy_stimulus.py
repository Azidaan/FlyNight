import numpy as np
import torch
from tracking.b1_index_copy_stimulus import indexed_stimulus


def test_distinct_index_copy_preserves_values_and_input_gradient():
    torch.manual_seed(26)
    for batch,time,types,columns in ((1,25,8,721),(2,3,2,7)):
        indices=np.arange(types*columns).reshape(types,columns)
        n=types*columns+3
        a=torch.randn(batch,time,1,columns,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        original=torch.zeros(batch,time,n)
        original[:,:,torch.as_tensor(indices)]=a.expand(-1,-1,types,-1)
        current=indexed_stimulus(torch,b,indices,n)
        assert torch.equal(original,current)
        weights=torch.randn_like(original)
        (original*weights).sum().backward()
        (current*weights).sum().backward()
        assert torch.allclose(a.grad,b.grad,rtol=1e-6,atol=1e-6)
