import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from tracking.z13_temporal_identity_readout import make_model, identity_loss


def inputs():
    torch.manual_seed(13)
    return (torch.randn(3, 4, 68), torch.randn(3, 68), torch.randn(3, 4, 14),
            torch.ones(3, 4, dtype=torch.bool), torch.randn(3, 3, 68),
            torch.tensor([[False, True, True]]*3), torch.randn(3, 4, 3, 5))


def test_zero_delta_exactly_reproduces_original_readout():
    model = make_model(torch)
    x, seed, geo, mask, px, pm, pg = inputs()
    actual, _, _ = model(x, seed, geo, mask, px, pm, pg)
    assert torch.equal(actual, model.base(x, seed, geo, mask))


def test_blank_history_is_invariant_even_after_nonzero_delta():
    model = make_model(torch, history=False)
    torch.nn.init.normal_(model.delta[-1].weight)
    x, seed, geo, mask, px, pm, pg = inputs()
    a, _, _ = model(x, seed, geo, mask, px, pm, pg)
    b, _, _ = model(x, seed, geo, mask, px*100, pm, pg*100)
    assert torch.equal(a, b) and torch.equal(a, model.base(x, seed, geo, mask))


def test_missing_identity_is_not_negative_supervision():
    logits = torch.tensor([[[2., -2.]]], requires_grad=True)
    targets = torch.tensor([[[1., 0.]]])
    known = torch.tensor([[[True, False]]])
    loss = identity_loss(torch, logits, targets, known, torch.ones_like(known))
    loss.backward()
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 0, 1] == 0


def test_geometry_control_is_invariant_to_all_activity():
    model = make_model(torch, activity=False)
    torch.nn.init.normal_(model.delta[-1].weight)
    x, seed, geo, mask, px, pm, pg = inputs()
    a, pair_a, _ = model(x, seed, geo, mask, px, pm, pg)
    b, pair_b, _ = model(x*10, seed*10, geo, mask, px*10, pm, pg)
    assert torch.equal(a, b) and torch.equal(pair_a, pair_b)


def test_null_action_and_padding_never_receive_identity_labels():
    model = make_model(torch)
    x, seed, geo, mask, px, pm, pg = inputs()
    mask[0, 3] = False
    score, logits, valid = model(x, seed, geo, mask, px, pm, pg)
    assert not valid[:, 0].any() and not valid[:, :, 0].any()
    assert not valid[0, 3].any() and score[0, 3] == -1e4
    loss = identity_loss(torch, logits, torch.zeros_like(logits), torch.zeros_like(valid), valid)
    loss.backward()
    assert loss == 0
