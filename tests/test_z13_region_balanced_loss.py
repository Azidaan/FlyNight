import os
os.environ['CUDA_VISIBLE_DEVICES']=''


def test_normalized_action_weights_match_explicit_loss_and_keep_gradient():
    import torch
    from tracking.z13_region_balanced_loss import balanced_action_loss
    torch.set_default_device('cpu')
    scores=torch.tensor([[1.,-1.],[.2,.3],[-.2,.1]],requires_grad=True)
    good=torch.tensor([[True,False],[False,False],[False,False]])
    mask=torch.ones_like(good)
    loss=balanced_action_loss(torch,scores,good,mask,2.)
    den=torch.logsumexp(torch.cat((scores,torch.zeros((3,1))),1),1)
    expected=((den[0]-scores[0,0])*2+den[1]+den[2])/4
    torch.testing.assert_close(loss,expected)
    loss.backward();assert bool(torch.isfinite(scores.grad).all())
    assert float(scores.grad[0,0])<0
