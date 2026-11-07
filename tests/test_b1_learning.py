import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from tracking.b1_learning import ReceptorHistory, identity_objective, apply_gate_policy, categorical_cost


def test_receptor_history_matches_original_numpy_definition():
    from scripts.z13_target_model import receptor_features_rich
    rng = np.random.default_rng(26)
    x = rng.random((2, 7, 11), dtype=np.float32)
    bb = ReceptorHistory('cpu'); state = bb.init(torch.from_numpy(x[:, :1]))
    expected = receptor_features_rich(x)
    for t in range(1, x.shape[1]):
        state, actual = bb.step(state, torch.from_numpy(x[:, t-1]), torch.from_numpy(x[:, t]))
        np.testing.assert_allclose(actual.numpy(), expected[t], atol=2e-7, rtol=2e-6)


def test_identity_loss_ignores_null_unknown_and_padding_with_live_gradient():
    p = torch.tensor([[float('nan'), .2, .8, .6]], dtype=torch.float64, requires_grad=True)
    delta = torch.zeros((1, 4), requires_grad=True)
    loss = identity_objective(p, delta, torch.tensor([[-1, 0, 1, -1]]), torch.tensor([[False, True, True, False]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert p.grad[0, 1] > 0 and p.grad[0, 2] < 0
    assert p.grad[0, 0] == p.grad[0, 3] == 0
    assert identity_objective(p, delta, torch.full((1, 4), -1), torch.ones((1, 4), dtype=torch.bool)) is None


def test_observable_policy_distinguishes_unchanged_null_and_real_region():
    geometry = np.zeros((3, 14)); geometry[2, -1] = 1
    assert apply_gate_policy([.4, .3, .99], geometry, .5) is None
    assert apply_gate_policy([.8, .3, .99], geometry, .5) == 0
    assert apply_gate_policy([.4, .9, .99], geometry, .5) == 1
    source = np.array([.34, .72], dtype=np.float32)
    assert np.array_equal(categorical_cost(source, [-1, 1, 0], None), source)
    assert np.array_equal(categorical_cost(source, [-1, 1, 0], 0), [1, 1])
    assert np.array_equal(categorical_cost(source, [-1, 1, 0], 1), [1, 0])
