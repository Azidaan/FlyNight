import numpy as np
import pytest
import torch
from tracking.b1_real_inputs import bind_observation, corrected_identity_probability


def data():
    archive = dict(ep_offset=np.array([0, 6]), ex_mask=np.array([[True, False, False]]),
                   fr_frame=np.arange(6), fr_crop=np.tile([0, 0, 3, 3], (6, 1)))
    row = dict(uid=1, video='fit', episode_local=0, current_array_index=5, current_frame=5,
               current_crop=[0, 0, 3, 3], retinal_source='source.npz', current_choices=[4], current_masks=[[1, 2]])
    return row, archive


def test_real_binding_preserves_null_and_observable_reference_and_ignores_truth():
    row, archive = data()
    first = bind_observation(row, archive, [-1, 4])
    archive['ep_identity'] = ['a deliberately wrong truth label']
    row['target_identity'] = 'another truth label'
    assert bind_observation(row, archive, [-1, 4]) == first
    assert first['reference_array_index'] == 3 and first['has_history']
    assert first['choices'] == [-1, 4] and first['null_slot'] == 0


def test_rejects_cross_episode_or_changed_candidate_bindings():
    row, archive = data()
    with pytest.raises(ValueError):
        bind_observation(row, archive, [-2, -1, 4])
    row['current_array_index'] = 6
    with pytest.raises(ValueError):
        bind_observation(row, archive, [-1, 4])


def test_probability_noop_is_exact_and_keeps_gradient_and_nan_semantics():
    p = torch.tensor([[float('nan'), .123456789, .987654321, float('nan')]], dtype=torch.float64)
    real = torch.tensor([[False, True, True, False]])
    delta = torch.zeros_like(p, requires_grad=True)
    out = corrected_identity_probability(p, delta, real)
    assert torch.equal(out[real], p[real]) and torch.equal(out.isnan(), p.isnan())
    out[real].sum().backward()
    assert (delta.grad[real] > 0).all() and torch.isfinite(delta.grad).all()
    shifted = corrected_identity_probability(p, torch.ones_like(p) * .3, real)
    assert torch.allclose(shifted[real], torch.sigmoid(torch.logit(p[real]) + .3))
