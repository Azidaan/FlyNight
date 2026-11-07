import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from tracking.b1_learning import ReceptorHistory
from tracking.b2_feature_temporal import TemporalDetectionBranch


def test_zero_branch_and_open_input_gradient_after_output_step():
    points = np.array([[2, 2], [2, 6], [6, 2], [6, 6]], np.float32)
    branch = TemporalDetectionBranch(3, points, 8).cpu()
    frames = [torch.rand(1, 3, 8, 8) for _ in range(5)]
    backend = ReceptorHistory('cpu')
    updated, delta = branch(frames, backend)
    assert torch.equal(updated, frames[-1]) and torch.count_nonzero(delta) == 0
    opt = torch.optim.SGD(branch.parameters(), lr=.1)
    updated.square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    updated, _ = branch(frames, backend); updated.square().mean().backward()
    assert torch.isfinite(branch.input.weight.grad).all() and branch.input.weight.grad.abs().sum() > 0


def test_detection_head_path_preserves_values_and_gradient():
    from tracking.b2_feature_temporal import detection_training_predictions
    class Head:
        end2end = True; one2one = dict(scale=2); one2many = dict(scale=3)
        def forward_head(self, features, scale): return features[0] * scale
    x = torch.ones(1, 3, requires_grad=True)
    y = detection_training_predictions(Head(), [x]); y.sum().backward()
    assert torch.equal(y, 2*x) and torch.equal(x.grad, torch.full_like(x, 2))
