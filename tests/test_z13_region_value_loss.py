import os
os.environ['CUDA_VISIBLE_DEVICES']=''


def test_signed_outcomes_reward_help_penalize_harm_and_leave_neutral_zero():
    import torch
    from tracking.z13_region_value_loss import signed_action_loss
    torch.set_default_device('cpu')
    scores=torch.zeros((1,4),requires_grad=True)
    good=torch.tensor([[True,False,False,False]])
    harm=torch.tensor([[False,True,False,False]])
    mask=torch.tensor([[True,True,True,False]])
    loss=signed_action_loss(torch,scores,good,harm,mask)
    torch.testing.assert_close(loss,torch.tensor(2/3))
    loss.backward()
    assert float(scores.grad[0,0])<0 and float(scores.grad[0,1])>0
    assert float(scores.grad[0,2])==float(scores.grad[0,3])==0
