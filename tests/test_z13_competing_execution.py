import torch
from tracking.z13_candidate_ranker import make_ranker
from tracking.z13_competing_candidates import marginal_choice_loss


def test_actual_ranker_trains_with_mixed_null_multiple_positive_and_padding_rows():
    torch.set_num_threads(2);torch.manual_seed(7);model=make_ranker(torch,True)
    x=torch.rand(128,3,68);s=torch.rand(128,68);g=torch.rand(128,3,8);mask=torch.arange(3)[None,:]<((torch.arange(128)%3)+1)[:,None]
    y=(torch.rand(128,3)>.8).float()*mask;opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    for _ in range(64):
        loss=marginal_choice_loss(torch,model(x,s,g,mask),y,mask,6.);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
