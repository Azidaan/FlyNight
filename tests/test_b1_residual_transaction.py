"""Historical engine rejects a failed step before publishing recurrent values."""
from types import SimpleNamespace

import pytest
import torch

from tracking.b1_learning import ReceptorHistory
from tracking.b1_tracker_runtime import ResidualEngine


class SmallBackend:
    """The state layouts of B/C/D without weights, files or GPU allocation."""
    def __init__(self, arm):
        self.arm = arm
        self.dev = torch.device('cpu')
        self.out = SimpleNamespace(w=torch.zeros(3, 2, 7))
        self.net = SimpleNamespace(stimulus=SimpleNamespace(n_nodes=5))
        self.index = torch.zeros((3, 4), dtype=torch.int64)

    def values(self, state):
        return state['h'] if self.arm == 'B' else state.nodes.activity

    def state(self, values):
        return {'h': values} if self.arm == 'B' else SimpleNamespace(nodes=SimpleNamespace(activity=values))

    def init(self, first):
        shape = (1, 2, 4) if self.arm == 'B' else (1, 5)
        return self.state(torch.full(shape, float(first.mean()), dtype=first.dtype))

    def step(self, state, previous, current):
        values = self.values(state) + current.mean() - previous.mean()
        return self.state(values), current[:, None].expand(-1, 3, -1).clone()


def engine(arm):
    obj = ResidualEngine.__new__(ResidualEngine)
    obj.arm = arm
    obj.bb = ReceptorHistory('cpu') if arm == 'R1' else SmallBackend(arm)
    return obj


def observations(dtype=torch.float32):
    frames = torch.arange(24, device='cpu', dtype=dtype).reshape(6, 4) / 100
    return frames, [dict(hex=x) for x in frames[:4]]


def state_values(obj, state):
    return state if obj.arm == 'R1' else obj.bb.values(state)


@pytest.mark.parametrize('arm', ['R1', 'B', 'C', 'D'])
@pytest.mark.parametrize('defect', ['state_nan', 'activity_nan', 'state_shape', 'activity_shape', 'state_device', 'activity_device', 'activity_none', 'exception'])
def test_backend_failure_keeps_eye_unchanged_and_clean_retry_matches_control(arm, defect):
    obj = engine(arm)
    frames, history = observations()
    eye, control = obj.start(history), obj.start(history)
    saved = dict(eye)
    saved_state = state_values(obj, eye['state']).clone()
    original = obj.bb.step

    def broken(*args):
        if defect == 'exception':
            raise RuntimeError('backend failure')
        state, activity = original(*args)
        values = state_values(obj, state)
        if defect == 'state_nan':
            values = torch.full_like(values, float('nan'))
        elif defect == 'state_shape':
            values = values[..., :-1]
        elif defect == 'state_device':
            values = torch.empty_like(values, device='meta')
        if defect.startswith('state_'):
            state = values if arm == 'R1' else obj.bb.state(values)
        if defect == 'activity_nan':
            activity = torch.full_like(activity, float('nan'))
        elif defect == 'activity_shape':
            activity = activity[..., :-1]
        elif defect == 'activity_device':
            activity = torch.empty_like(activity, device='meta')
        elif defect == 'activity_none':
            activity = None
        return state, activity

    obj.bb.step = broken
    with pytest.raises((ValueError, RuntimeError)):
        obj.advance(eye, frames[4])
    assert eye.keys() == saved.keys() and all(eye[k] is saved[k] for k in saved)
    assert torch.equal(state_values(obj, eye['state']), saved_state)
    obj.bb.step = original
    obj.advance(eye, frames[4]); obj.advance(control, frames[4])
    assert torch.equal(state_values(obj, eye['state']), state_values(obj, control['state']))
    assert torch.equal(eye['current'], control['current'])
    assert torch.equal(eye['last'], control['last'])


@pytest.mark.parametrize('arm', ['R1', 'B', 'C', 'D'])
@pytest.mark.parametrize('defect', ['initial_state', 'history_state', 'history_activity'])
def test_invalid_initial_history_is_rejected_and_can_be_retried(arm, defect):
    obj = engine(arm)
    _, history = observations()
    expected = obj.start(history)
    original_init, original_step = obj.bb.init, obj.bb.step

    def invalid_state(state):
        values = torch.full_like(state_values(obj, state), float('nan'))
        return values if arm == 'R1' else obj.bb.state(values)

    if defect == 'initial_state':
        obj.bb.init = lambda x: invalid_state(original_init(x))
    else:
        calls = 0
        def broken(*args):
            nonlocal calls
            calls += 1
            state, activity = original_step(*args)
            # A later valid state must not conceal an invalid intermediate step.
            if calls == 1:
                if defect == 'history_state':
                    state = invalid_state(state)
                else:
                    activity = torch.full_like(activity, float('nan'))
            return state, activity
        obj.bb.step = broken
    with pytest.raises(ValueError):
        obj.start(history)
    obj.bb.init, obj.bb.step = original_init, original_step
    actual = obj.start(history)
    assert torch.equal(state_values(obj, actual['state']), state_values(obj, expected['state']))
    assert torch.equal(actual['reference'], expected['reference'])


@pytest.mark.parametrize('bad', ['nan', 'shape', 'dtype', 'device'])
def test_invalid_input_rejected_before_backend_call(bad):
    obj = engine('R1')
    frames, history = observations()
    eye = obj.start(history)
    value = frames[4]
    if bad == 'nan': value = torch.full_like(value, float('nan'))
    elif bad == 'shape': value = value[:-1]
    elif bad == 'dtype': value = value.double()
    else: value = torch.empty_like(value, device='meta')
    def forbidden(*args): raise AssertionError('Invalid input reached backend')
    obj.bb.step = forbidden
    with pytest.raises(ValueError): obj.advance(eye, value)


def test_valid_float64_history_and_observation_are_not_cast():
    obj = engine('R1')
    frames, history = observations(torch.float64)
    eye = obj.start(history)
    obj.advance(eye, frames[4])
    assert eye['state'].dtype == eye['current'].dtype == eye['last'].dtype == torch.float64
