import numpy as np
import torch
from tracking.z13_competing_candidates import marginal_choice_loss,quota_precision


def test_single_good_choice_is_exact_four_way_cross_entropy():
    s=torch.tensor([[1.,2.,3.],[2.,-1.,.2]],requires_grad=True);m=torch.ones_like(s,dtype=torch.bool);y=torch.tensor([[0.,1.,0.],[0.,0.,0.]])
    a=marginal_choice_loss(torch,s,y,m,1.)
    b=torch.nn.functional.cross_entropy(torch.cat([s,torch.zeros(2,1)],1),torch.tensor([1,3]))
    assert torch.equal(a,b);a.backward();assert torch.isfinite(s.grad).all()


def test_multiple_acceptable_choices_and_padding_do_not_invent_a_winner():
    s=torch.tensor([[1.,2.,99.]],requires_grad=True);m=torch.tensor([[True,True,False]]);y=torch.tensor([[1.,1.,0.]])
    expected=torch.logsumexp(torch.tensor([1.,2.,0.]),0)-torch.logsumexp(torch.tensor([1.,2.]),0)
    a=marginal_choice_loss(torch,s,y,m,1.);assert torch.equal(a,expected)
    assert torch.equal(a,marginal_choice_loss(torch,s[:,[1,0,2]],y[:,[1,0,2]],m[:,[1,0,2]],1.))
    a.backward();assert s.grad[0,2]==0 and torch.isfinite(s.grad).all()


def test_null_rows_push_scores_down_and_unknown_is_not_a_measured_error():
    s=torch.zeros(1,3,requires_grad=True);m=torch.tensor([[True,False,False]])
    loss=marginal_choice_loss(torch,s,torch.zeros_like(s),m,3.);loss.backward();assert s.grad[0,0]>0 and s.grad[0,1]==0
    rows=[dict(video='a',p_present=2.,kind=-1),dict(video='a',p_present=1.,kind=0,useful=False)]
    pv,counts=quota_precision(rows,q=.5);assert pv=={} and counts['a']['unknown']==1 and counts['a']['known_nonuseful']==0
