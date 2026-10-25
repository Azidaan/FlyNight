import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from scripts.z13_temporal_identity_vectorized_recovery2_20261025 import policy,loss_function
from scripts.z13_temporal_identity_vectorized_recovery3_20261025 import structural_stack
from tracking.z13_region_assignment import apply_policy
from tracking.z13_temporal_identity_readout import make_model,identity_loss
from tracking.z13_region_value_loss import signed_action_loss


def test_grouped_policy_is_bit_exact_for_variable_candidate_counts_and_ties():
    rng=np.random.default_rng(15)
    logits=[rng.normal(size=k).astype(np.float32) for k in range(1,17) for _ in range(5)]
    logits.extend([np.zeros(5),np.ones(3),np.array([-1.,-2.])])
    index,prob,positive=policy(logits)
    for tau in (0.,.1,.5,.9,1.01):
        expected=[apply_policy(x,tau) for x in logits]
        assert [a for a,b in expected]==np.where(positive&(prob>=tau),index,-1).tolist()
        assert np.array_equal(np.array([b for a,b in expected]),prob)


def test_vectorized_loss_and_gradients_preserve_independent_heads():
    import copy
    torch.manual_seed(3);models=[make_model(torch),make_model(torch)]
    b=dict(x=torch.randn(4,3,68),seed=torch.randn(4,68),geo=torch.randn(4,3,14),mask=torch.ones(4,3,dtype=torch.bool),
        px=torch.randn(4,2,68),pm=torch.tensor([[False,True]]*4),pg=torch.randn(4,3,2,5),
        good=torch.zeros(4,3,dtype=torch.bool),harm=torch.ones(4,3,dtype=torch.bool),id_target=torch.zeros(4,3,2),id_known=torch.ones(4,3,2,dtype=torch.bool))
    expected=[]
    for m in models:
        s,a,v=m(*[b[k] for k in ('x','seed','geo','mask','px','pm','pg')])
        loss=signed_action_loss(torch,s,b['good'],b['harm'],b['mask'])+identity_loss(torch,a,b['id_target'],b['id_known'],v)
        expected.append((float(loss.detach()),torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True)))
    params,buffers=structural_stack(models,torch);template=copy.deepcopy(models[0]).to('meta')
    grads,values=torch.func.vmap(torch.func.grad_and_value(loss_function(torch,template,'D')),in_dims=(0,0,0))(params,buffers,{k:torch.stack([v,v]) for k,v in b.items()})
    for i,(value,grad) in enumerate(expected):
        assert abs(float(values[i])-value)<1e-6
        for (name,param),g in zip(models[i].named_parameters(),grad):
            assert torch.allclose(grads[name][i],torch.zeros_like(param) if g is None else g,atol=1e-6,rtol=1e-5)
