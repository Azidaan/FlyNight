import torch
from tracking.z13_region_semantic_loss import semantic_choice_loss
from tracking.z13_competing_candidates import marginal_choice_loss


def test_utility_special_case_matches_old_loss_and_gradients():
    scores=torch.tensor([[.2,-.4,.7],[-.1,.3,.5],[.9,-.2,-.3]],requires_grad=True)
    target=torch.tensor([[True,False,False],[False,False,False],[False,True,True]])
    mask=torch.tensor([[True,True,False],[True,True,True],[True,True,True]])
    old=marginal_choice_loss(torch,scores,target,mask,1.);old_grad=torch.autograd.grad(old,scores,retain_graph=True)[0]
    new=semantic_choice_loss(torch,scores,target,~(target&mask).any(1),mask);new_grad=torch.autograd.grad(new,scores)[0]
    torch.testing.assert_close(old,new,rtol=1e-6,atol=1e-7);torch.testing.assert_close(old_grad,new_grad,rtol=1e-6,atol=1e-7)


def test_correct_baseline_is_acceptable_without_forcing_neutral_intervention():
    scores=torch.tensor([[-5.,-7.]],requires_grad=True);mask=torch.ones_like(scores,dtype=torch.bool)
    loss=semantic_choice_loss(torch,scores,torch.tensor([[True,False]]),torch.tensor([True]),mask)
    assert float(loss.detach())<.001
    loss.backward();assert scores.grad[0,1]>0 and torch.isfinite(scores.grad).all()


def test_unrecoverable_and_padded_actions_have_finite_null_fallback_gradient():
    scores=torch.tensor([[.3,-.1,1e4]],requires_grad=True);mask=torch.tensor([[True,True,False]])
    loss=semantic_choice_loss(torch,scores,torch.zeros_like(mask),torch.tensor([False]),mask)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(scores.grad).all() and scores.grad[0,2]==0
