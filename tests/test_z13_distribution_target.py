import torch
import pytest
from tracking.z13_distribution_target import spatial_targets, distribution_loss


def test_known_absent_and_unknown_are_distinct():
    points = torch.tensor([[0., 0.], [1., 0.], [3., 0.]])
    target, mask = spatial_targets(torch, [1, 0, -1], [[0., 0.], [float('nan'), 0], [float('nan'), 0]], points, 1.)
    assert torch.allclose(target.sum(1), torch.tensor([1., 1., 0.]))
    assert target[0, 0] > target[0, 1] > target[0, 2] > target[0, 3]
    assert target[1, -1] == 1 and mask.tolist() == [True, True, False]
    logits = torch.randn(3, 4, requires_grad=True)
    distribution_loss(torch, logits, target, mask).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[2].count_nonzero() == 0


def test_geometry_translation_scale_and_boundary_normalization():
    points = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    center = torch.tensor([[-.5, 0.]])
    first, _ = spatial_targets(torch, [1], center, points, 1.)
    second, _ = spatial_targets(torch, [1], center * 3 + 7, points * 3 + 7, 3.)
    assert torch.allclose(first, second, atol=1e-7)
    assert torch.allclose(first.sum(), torch.tensor(1.))


def test_near_miss_gets_lower_loss_than_far_miss():
    points = torch.tensor([[0., 0.], [1., 0.], [4., 0.]])
    target, known = spatial_targets(torch, [1], [[0., 0.]], points, 1.)
    near = torch.tensor([[0., 4., 0., 0.]])
    far = torch.tensor([[0., 0., 4., 0.]])
    assert distribution_loss(torch, near, target, known) < distribution_loss(torch, far, target, known)
    # The original one-hot objective cannot distinguish these two mistakes.
    assert torch.nn.functional.cross_entropy(near, torch.tensor([0])) == torch.nn.functional.cross_entropy(far, torch.tensor([0]))


def test_reject_invalid_scale():
    with pytest.raises(ValueError):
        spatial_targets(torch, [1], [[0., 0.]], torch.zeros(1, 2), 0.)
