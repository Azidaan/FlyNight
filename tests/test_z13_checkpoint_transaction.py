"""A rejected resume must leave weights, anchors, optimizer and RNG intact."""
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracking.z13_checkpoint import restore_checkpoint, save_checkpoint


def equivalent(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equivalent(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equivalent(a, b)
    else:
        assert left == right


def snapshot(core, readout, optimizer, rng):
    return deepcopy(dict(core=core.net.state_dict(), readout=readout.state_dict(),
                         anchors=core.p0, optimizer=optimizer.state_dict(),
                         rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                         rng_torch=torch.get_rng_state(), generator=rng.bit_generator.state))


def fixture_checkpoint(tmp_path):
    core = SimpleNamespace(net=torch.nn.Linear(2, 2, device='cpu'))
    core.p0 = [value.detach().clone() for value in core.net.parameters()]
    readout = torch.nn.Linear(2, 2, device='cpu')
    optimizer = torch.optim.Adam(list(readout.parameters()) + list(core.net.parameters()), lr=.003)
    rng = np.random.default_rng(49)
    optimizer.zero_grad()
    (core.net(torch.ones(2, device='cpu')) + readout(torch.ones(2, device='cpu'))).square().sum().backward()
    optimizer.step()
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(torch, path, 'D', core, readout, optimizer, rng, 1, [{'it': 1}], {'fixture': 1})
    saved = torch.load(path, map_location='cpu', weights_only=False)
    with torch.no_grad():
        for p in list(readout.parameters()) + list(core.net.parameters()) + core.p0:
            p.add_(3)
    optimizer.param_groups[0]['lr'] = .007
    return core, readout, optimizer, rng, path, saved


def damage(state, kind):
    if kind == 'backbone_key':
        del state['backbone']['bias']
    elif kind == 'backbone_overflow':
        state['backbone']['weight'] = torch.full((2, 2), 1e300, dtype=torch.float64, device='cpu')
    elif kind == 'anchor_count':
        state['anchors'] = []
    elif kind == 'anchor_shape':
        state['anchors'][0] = torch.ones(1, device='cpu')
    elif kind == 'anchor_overflow':
        state['anchors'][0] = torch.full((2, 2), 1e300, dtype=torch.float64, device='cpu')
    elif kind == 'optimizer_shape':
        state['optimizer']['state'][0]['exp_avg'] = torch.zeros(17, device='cpu')
    elif kind == 'optimizer_overflow':
        state['optimizer']['state'][0]['exp_avg'] = torch.full((2, 2), 1e300, dtype=torch.float64, device='cpu')
    elif kind == 'python_rng':
        state['rng_python'] = (3, (), None)
    elif kind == 'numpy_rng':
        state['rng_numpy'] = ('invalid',)
    elif kind == 'torch_rng':
        state['rng_torch'] = torch.zeros(1, dtype=torch.uint8, device='cpu')
    elif kind == 'generator_rng':
        state['generator'] = {'bit_generator': 'invalid'}
    elif kind == 'step_missing':
        del state['step']
    elif kind == 'step_boolean':
        state['step'] = True
    elif kind == 'history_type':
        state['history'] = None


@pytest.mark.parametrize('kind', ['backbone_key', 'backbone_overflow', 'anchor_count',
    'anchor_shape', 'anchor_overflow', 'optimizer_shape', 'optimizer_overflow',
    'python_rng', 'numpy_rng', 'torch_rng', 'generator_rng', 'step_missing',
    'step_boolean', 'history_type'])
def test_invalid_resume_is_transactional(tmp_path, kind):
    core, readout, optimizer, rng, path, state = fixture_checkpoint(tmp_path)
    damage(state, kind)
    torch.save(state, path)
    before = snapshot(core, readout, optimizer, rng)
    with pytest.raises((ValueError, RuntimeError, KeyError, TypeError)):
        restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
    equivalent(before, snapshot(core, readout, optimizer, rng))


def test_live_optimizer_hook_failure_rolls_back_without_reinvoking_hook(tmp_path):
    core, readout, optimizer, rng, path, _ = fixture_checkpoint(tmp_path)
    before = snapshot(core, readout, optimizer, rng)
    parameters = list(optimizer.param_groups[0]['params'])
    calls = []
    def hook(actual, state):
        if actual is optimizer:
            actual.param_groups[0]['lr'] = 100
            random.random(); np.random.random(); torch.rand(1, device='cpu')
            calls.append(1)
            raise RuntimeError('live optimizer hook failed')
    optimizer.register_load_state_dict_pre_hook(hook)
    with pytest.raises(RuntimeError, match='live optimizer'):
        restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
    equivalent(before, snapshot(core, readout, optimizer, rng))
    assert len(calls) == 1
    assert all(a is b for a, b in zip(parameters, optimizer.param_groups[0]['params']))


def test_live_model_hook_failure_restores_already_copied_weights(tmp_path):
    core, readout, optimizer, rng, path, _ = fixture_checkpoint(tmp_path)
    before = snapshot(core, readout, optimizer, rng)
    def fail(module, incompatible):
        raise RuntimeError('model load hook failed')
    readout.register_load_state_dict_post_hook(fail)
    with pytest.raises(RuntimeError, match='model load hook'):
        restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
    equivalent(before, snapshot(core, readout, optimizer, rng))


def test_staging_hook_failure_does_not_consume_random_streams(tmp_path):
    core, readout, optimizer, rng, path, _ = fixture_checkpoint(tmp_path)
    before = snapshot(core, readout, optimizer, rng)
    def fail(actual, state):
        random.random(); np.random.random(); torch.rand(1, device='cpu'); rng.random()
        raise RuntimeError('staging hook failed')
    optimizer.register_load_state_dict_pre_hook(fail)
    with pytest.raises(RuntimeError, match='staging hook'):
        restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
    equivalent(before, snapshot(core, readout, optimizer, rng))


def test_gpu_checkpoint_requires_explicit_cpu_transfer_without_initializing_cuda(tmp_path, monkeypatch):
    core, readout, optimizer, rng, path, saved = fixture_checkpoint(tmp_path)
    saved['rng_cuda'] = [torch.tensor([1, 2, 3], dtype=torch.uint8, device='cpu')]
    torch.save(saved, path)
    before = snapshot(core, readout, optimizer, rng)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    def forbidden(*a, **kw):
        raise AssertionError('CPU checkpoint restore touched CUDA generators')
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    with pytest.raises(ValueError, match='restore_cuda_rng=False'):
        restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
    equivalent(before, snapshot(core, readout, optimizer, rng))
    step, history = restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng,
                                       {'fixture': 1}, restore_cuda_rng=False)
    assert step == 1 and history == [{'it': 1}]
    equivalent(saved['backbone'], core.net.state_dict())
    equivalent(saved['readout'], readout.state_dict())


@pytest.mark.parametrize('mode', ['success', 'count_mismatch', 'live_failure'])
def test_initialized_cuda_contract_with_private_mock_generators(tmp_path, monkeypatch, mode):
    # Control-flow test only: no CUDA kernels, allocation or actual device RNG.
    core, readout, optimizer, rng, path, saved = fixture_checkpoint(tmp_path)
    target = [torch.tensor([7, 8, 9], dtype=torch.uint8, device='cpu')]
    saved['rng_cuda'] = target * (2 if mode == 'count_mismatch' else 1)
    torch.save(saved, path)
    current = [torch.tensor([1, 2, 3], dtype=torch.uint8, device='cpu')]
    original = deepcopy(current)
    real_generator = torch.Generator
    class CudaGenerator:
        def set_state(self, value):
            assert value.device.type == 'cpu' and value.dtype == torch.uint8 and value.shape == (3,)
    monkeypatch.setattr(torch, 'Generator', lambda device: CudaGenerator() if str(device).startswith('cuda') else real_generator(device=device))
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: deepcopy(current))
    def assign(values):
        current[:] = deepcopy(values)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', assign)
    before = snapshot(core, readout, optimizer, rng)
    if mode == 'live_failure':
        def fail(actual, state):
            if actual is optimizer:
                current[0].zero_()
                raise RuntimeError('GPU hook failure')
        optimizer.register_load_state_dict_pre_hook(fail)
    if mode != 'success':
        with pytest.raises((RuntimeError, ValueError)):
            restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})
        equivalent(before, snapshot(core, readout, optimizer, rng))
        equivalent(original, current)
    else:
        assert restore_checkpoint(torch, path, 'D', core, readout, optimizer, rng, {'fixture': 1})[0] == 1
        equivalent(target, current)
