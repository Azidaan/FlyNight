import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from tracking.b1_fast_stimulus import assign_zero_stimulus


def test_zero_buffer_assignment_matches_original_add_exactly():
    for batch,steps,receptors,columns in [(1,25,8,721),(2,3,2,7)]:
        index=np.arange(receptors*columns).reshape(receptors,columns)
        frames=torch.rand(batch,steps,columns)
        old=torch.zeros(batch,steps,receptors*columns+3)
        new=torch.zeros_like(old)
        old[:,:,index]+=frames[:,:,None]
        assign_zero_stimulus(torch,new,frames,index)
        assert torch.equal(old,new)
