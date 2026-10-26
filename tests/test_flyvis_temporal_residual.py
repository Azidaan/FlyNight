import torch

from tracking.flyvis_temporal_residual import TemporalResidualAssociation


def sample():
    torch.manual_seed(17)
    head = TemporalResidualAssociation(channels=4, hidden=8)
    reference, current = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
    masks = torch.zeros(2, 3, 7, dtype=torch.bool)
    masks[:, 1, :3] = True
    masks[:, 2, 3:] = True
    args = (torch.randn(2, 3), reference, current, masks[:, 1], masks,
            torch.ones(2, 3, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    return head, args


def test_zero_start_preserves_all_base_scores_exactly():
    head, args = sample()
    scores, correction = head(*args)
    assert torch.equal(scores, args[0]) and not correction.any()


def test_disabled_history_empty_reference_and_invalid_slots_cannot_change_base():
    head, args = sample()
    torch.nn.init.normal_(head.output.weight)
    missing = list(args)
    missing[-1] = torch.zeros_like(args[-1])
    assert torch.equal(head(*missing)[0], args[0])
    missing = list(args)
    missing[3] = torch.zeros_like(args[3])
    assert torch.equal(head(*missing)[0], args[0])
    missing = list(args)
    missing[-2] = torch.zeros_like(args[-2])
    assert torch.equal(head(*missing)[0], args[0])


def test_candidate_permutation_does_not_change_pair_meaning_and_null_is_unchanged():
    head, args = sample()
    torch.nn.init.normal_(head.output.weight)
    perm = torch.tensor([0, 2, 1])
    original, _ = head(*args)
    changed = list(args)
    for index in (0, 4, 5):
        changed[index] = changed[index][:, perm]
    result, _ = head(*changed)
    assert torch.allclose(result, original[:, perm])
    assert torch.equal(result[:, 0], changed[0][:, 0])


def test_head_opens_gradient_path_after_first_output_step():
    head, args = sample()
    args = list(args)
    args[1].requires_grad_()
    args[2].requires_grad_()
    optimizer = torch.optim.SGD(head.output.parameters(), lr=.1)
    loss = torch.nn.functional.cross_entropy(head(*args)[0], torch.tensor([1, 2]))
    loss.backward()
    assert head.output.weight.grad.norm() > 0
    optimizer.step()
    head.zero_grad(set_to_none=True)
    args[1].grad = args[2].grad = None
    torch.nn.functional.cross_entropy(head(*args)[0], torch.tensor([1, 2])).backward()
    assert args[1].grad.norm() > 0 and args[2].grad.norm() > 0
