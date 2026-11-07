import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pytest
import torch
from tracking.b2_causal_detector import FeatureRing, CausalP3Branch, detection_targets, CrowdAwareBCE


class Backend:
    def __init__(self):
        self.dt = .02; self.steps = 10; self.intervals = []
    def init(self, first):
        return first[:, 0]
    def step(self, state, previous, current):
        self.intervals.append(self.dt * self.steps)
        state = .5 * state + current
        return state, state[:, None].expand(-1, 34, -1)


def test_ring_cannot_retain_future_or_more_than_five_features():
    ring = FeatureRing()
    for i in range(8): ring.append(i, [torch.full((1, 2, 2, 2), float(i), requires_grad=True)])
    seq, frames = ring.sequence()
    assert frames == [3, 4, 5, 6, 7] and all(not x.requires_grad for x in seq)
    assert ring.bytes == 5 * 8 * 4
    with pytest.raises(ValueError): ring.append(7, [seq[-1]])


def test_last_control_ignores_earlier_features_mean_retains_them_and_real_intervals():
    torch.manual_seed(42)
    branch = CausalP3Branch(2, np.array([[0., 0.], [1., 1.]], np.float32), 2)
    torch.nn.init.constant_(branch.output.weight, .1)
    sequence = [torch.full((1, 2, 3, 4), i / 5) for i in range(5)]
    frames = [0, 1, 7, 13, 19]
    a = Backend(); last, _ = branch(sequence, a, frames, 30, 'last')
    changed = [x + 10 for x in sequence[:-1]] + sequence[-1:]
    last2, _ = branch(changed, Backend(), frames, 30, 'last')
    mean, _ = branch(sequence, Backend(), frames, 30, 'mean')
    mean2, _ = branch(changed, Backend(), frames, 30, 'mean')
    assert torch.equal(last, last2) and not torch.equal(mean, mean2)
    assert a.intervals == pytest.approx([1/30, .2, .2, .2])
    with pytest.raises(ValueError): branch(sequence, a, [0, 1, 7, 13, 12], 30)


def test_letterbox_targets_preserve_crowd_as_ignore_and_clip_boxes():
    entry = {'labels': [dict(category='car', box2d=dict(x1=0, y1=0, x2=1280, y2=720)),
                        dict(category='car', box2d=dict(x1=10, y1=10, x2=20, y2=30), attributes={'crowd': True})]}
    out = detection_targets(entry, {0: 'car'}, (720, 1280), (544, 960))
    np.testing.assert_allclose(out['bboxes'], [[.5, .5, 1, 540/544]], atol=1e-7)
    assert out['padding'] == [0, 2] and out['crowd_xyxy'].shape == (1, 4)


def test_crowd_negative_loss_is_zero_but_positive_and_other_class_gradients_remain():
    centers = torch.tensor([[5., 5.], [20., 20.]])
    bce = CrowdAwareBCE(centers, torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0]), 2)
    x = torch.zeros(1, 2, 2, requires_grad=True)
    y = torch.zeros_like(x)
    loss = bce(x, y); loss.sum().backward()
    assert x.grad[0, 0, 0] == 0 and x.grad[0, 0, 1] > 0 and x.grad[0, 1, 0] > 0
    y[0, 0, 0] = 1
    assert bce(x, y)[0, 0, 0] > 0
