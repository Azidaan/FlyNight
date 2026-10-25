import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from tracking.z13_default_class import action_loss, make_model


def test_all_neutral_actions_train_toward_default():
    z = torch.tensor([[.2, .3]], requires_grad=True)
    zero = torch.zeros((1, 2), dtype=torch.bool); mask = ~zero
    action_loss(torch, z, zero, zero, mask, 'cost_ce', .01).backward()
    assert (z.grad > 0).all()  # gradient descent lowers both neutral actions


def test_helpful_action_above_harmful_and_default():
    z = torch.zeros((1, 2), requires_grad=True)
    good = torch.tensor([[True, False]]); harm = ~good
    action_loss(torch, z, good, harm, torch.ones_like(good), 'cost_ce', .01).backward()
    assert z.grad[0, 0] < 0 and z.grad[0, 1] > 0


def test_no_fly_control_cannot_use_activity_or_target():
    torch.manual_seed(0); model = make_model(torch, activity=False)
    x = torch.randn(2, 3, 68); seed = torch.randn(2, 68)
    geo = torch.randn(2, 3, 14); mask = torch.ones(2, 3, dtype=torch.bool)
    assert torch.equal(model(x, seed, geo, mask), model(x*19, seed+20, geo, mask))


def test_padding_does_not_change_loss():
    z = torch.tensor([[.2, -.4]]); good = torch.tensor([[True, False]]); harm = ~good
    valid = torch.ones_like(good)
    before = action_loss(torch, z, good, harm, valid, 'cost_ce', .01)
    after = action_loss(torch, torch.cat([z, torch.tensor([[100.]])], 1),
                        torch.cat([good, torch.tensor([[True]])], 1),
                        torch.cat([harm, torch.tensor([[False]])], 1),
                        torch.tensor([[True, True, False]]), 'cost_ce', .01)
    assert torch.equal(before, after)
