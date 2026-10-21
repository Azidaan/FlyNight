import torch
from tracking.z13_pairwise_candidate_loss import pairwise_candidate_loss


def test_preferred_candidate_gradient_and_score_shift_invariance():
    scores=torch.tensor([[0.,0.,0.]],requires_grad=True);target=torch.tensor([[1.,0.,0.]]);mask=torch.tensor([[True,True,False]])
    loss,count=pairwise_candidate_loss(torch,scores,target,mask);assert count==1
    loss.backward();assert scores.grad[0,0]<0 and scores.grad[0,1]>0 and scores.grad[0,2]==0
    shifted,_=pairwise_candidate_loss(torch,scores.detach()+123.,target,mask);assert torch.allclose(loss,shifted)
    ordered,_=pairwise_candidate_loss(torch,torch.tensor([[3.,-3.,999.]]),target,mask);assert ordered<loss


def test_padding_and_equal_labels_do_not_create_training_pairs():
    target=torch.tensor([[1.,1.,0.]]);mask=torch.tensor([[True,True,False]])
    loss,count=pairwise_candidate_loss(torch,torch.tensor([[10.,-10.,1e4]],requires_grad=True),target,mask)
    assert count==0 and loss==0 and torch.isfinite(loss)


def test_empty_pair_batch_has_finite_zero_gradient():
    scores=torch.randn(4,3,requires_grad=True);target=torch.zeros_like(scores);mask=torch.ones_like(scores,dtype=torch.bool)
    loss,count=pairwise_candidate_loss(torch,scores,target,mask);loss.backward()
    assert count==0 and torch.isfinite(loss) and torch.count_nonzero(scores.grad)==0
