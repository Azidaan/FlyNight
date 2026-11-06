"""Failed neural observations must not commit poisoned recurrent state."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracking.flyvis_motion import FlyVisMotionEstimator


def model():
    value=FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
    value.torch=torch
    value.flyvis=SimpleNamespace(device=torch.device('cpu'))
    value.eye_width=value.eye_height=391
    value.kernel_area=169
    value.flow_gain=1.0
    value.geometry_version='legacy_v1'
    value._local_centers_yx=np.array([[0,0],[10,10]],np.float32)
    value.integration_steps=1
    value.dt=.02
    value.temporal_resampling='hold'
    value.flow_aggregation='mean'
    value.subtract_initial_baseline=True
    value.stats=dict(observations=0,render_seconds=0.,network_seconds=0.,decoder_seconds=0.)
    value.reset()
    value._bounds=[(0,0,1280,720)]
    value._frame_size=(1280,720)
    value._timed=lambda fn:(fn(),0.)
    value._render=lambda frame:torch.tensor([[[[float(np.asarray(frame).flat[0])]*2]]])
    value.bad_initial=False
    value.bad_activity=False
    value.decoder_mode='valid'
    def state(activity):
        return SimpleNamespace(nodes=SimpleNamespace(activity=activity))
    value._initial_state=lambda rendered:state(torch.full((1,3),float('nan') if value.bad_initial else 1.))
    def simulate(stimulus,dt,initial_state,as_states):
        activity=initial_state.nodes.activity+stimulus.mean()
        if value.bad_activity:
            activity=activity*float('nan')
        return [state(activity)]
    value.network=SimpleNamespace(simulate=simulate)
    def decoder(activity):
        if value.decoder_mode=='raise':
            raise RuntimeError('injected decoder failure')
        if value.decoder_mode=='nan':
            return torch.full((1,1,2,2),float('nan'))
        if value.decoder_mode=='wrong_shape':
            return torch.zeros(1,1,3,2)
        if value.decoder_mode=='overflow':
            return torch.full((1,1,2,2),torch.finfo(torch.float32).max)
        # A finite decoder output can conceal a nonfinite neural state.
        return torch.ones(1,1,2,2)*torch.nan_to_num(activity).mean()
    value.decoder=decoder
    return value


@pytest.mark.parametrize('failure',['baseline_nan','baseline_shape','initial_activity_nan'])
def test_failed_first_observation_leaves_uninitialized_state(failure):
    estimator=model()
    if failure=='baseline_nan':estimator.decoder_mode='nan'
    if failure=='baseline_shape':estimator.decoder_mode='wrong_shape'
    if failure=='initial_activity_nan':estimator.bad_initial=True
    with pytest.raises(ValueError):
        estimator.observe(np.array([[1]],np.float32))
    assert estimator._state is None
    assert estimator._last_hex is None
    assert estimator._baseline_flow is None
    assert estimator.baseline_activity is None
    estimator.decoder_mode='valid'; estimator.bad_initial=False
    actual=estimator.observe(np.array([[1]],np.float32))
    expected=model().observe(np.array([[1]],np.float32))
    np.testing.assert_array_equal(actual.vectors_xy,expected.vectors_xy)


@pytest.mark.parametrize('failure',['decoder_nan','activity_nan','decoder_exception'])
def test_failed_later_observation_preserves_last_successful_frame(failure):
    estimator=model(); control=model()
    frame=np.array([[1]],np.float32)
    estimator.observe(frame); control.observe(frame)
    old_state=estimator._state
    old_hex=estimator._last_hex.clone()
    old_baseline=estimator._baseline_flow.clone()
    if failure=='decoder_nan':estimator.decoder_mode='nan'
    if failure=='activity_nan':estimator.bad_activity=True
    if failure=='decoder_exception':estimator.decoder_mode='raise'
    with pytest.raises((ValueError,RuntimeError)):
        estimator.observe(np.array([[8]],np.float32))
    assert estimator._state is old_state
    assert torch.equal(estimator._last_hex,old_hex)
    assert torch.equal(estimator._baseline_flow,old_baseline)
    assert estimator.last_decoded is None and estimator.last_activity is None
    estimator.decoder_mode='valid'; estimator.bad_activity=False
    actual=estimator.observe(np.array([[3]],np.float32))
    expected=control.observe(np.array([[3]],np.float32))
    np.testing.assert_array_equal(actual.points_xy,expected.points_xy)
    np.testing.assert_array_equal(actual.vectors_xy,expected.vectors_xy)
    assert torch.equal(estimator._state.nodes.activity,control._state.nodes.activity)


@pytest.mark.parametrize('failure',['nan','wrong_shape','overflow'])
def test_failed_observation_preserves_previous_published_diagnostics(failure):
    estimator=model(); control=model()
    for level in (1,2):
        frame=np.array([[level]],np.float32)
        estimator.observe(frame); control.observe(frame)
    previous={name:getattr(estimator,name) for name in
        ('_state','_last_hex','last_decoded','last_activity','_baseline_flow','baseline_activity')}
    estimator.decoder_mode=failure
    with np.errstate(over='ignore',invalid='ignore'), pytest.raises((ValueError,RuntimeError)):
        estimator.observe(np.array([[8]],np.float32))
    for name,value in previous.items():
        assert getattr(estimator,name) is value,name
    estimator.decoder_mode='valid'
    field=estimator.observe(np.array([[3]],np.float32))
    expected=control.observe(np.array([[3]],np.float32))
    np.testing.assert_array_equal(field.vectors_xy,expected.vectors_xy)
    for aggregation in ('mean','last','sum'):
        np.testing.assert_array_equal(estimator.field_from_last(aggregation).vectors_xy,
                                      control.field_from_last(aggregation).vectors_xy)
