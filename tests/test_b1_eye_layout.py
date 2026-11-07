import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import pytest
from tracking.b1_eye_layout import canonical_eye_layout
from tracking.b1_learning import ReceptorHistory
from tracking.flyvis_temporal_residual import causal_pair_activities


def test_extra_unit_channel_is_removed_without_changing_receptors():
    raw = torch.arange(5 * 721, dtype=torch.float32).reshape(5, 1, 1, 721)
    canonical = canonical_eye_layout(raw, 5, 721)
    assert canonical.shape == (5, 1, 721)
    assert torch.equal(canonical.flatten(), raw.flatten())
    reference, current = causal_pair_activities(ReceptorHistory('cpu'), canonical[:, 0][None], 3)
    assert reference.shape == current.shape == (1, 34, 721)
    # В прежнем живом пути R1 дополнительная ось попадала в broadcast.
    bb = ReceptorHistory('cpu')
    h = raw[:, 0][None]
    state = bb.init(h[:, :1])
    _, wrong = bb.step(state, h[:, 0], h[:, 1])
    assert wrong.shape != (1, 34, 721)


def test_multiple_channels_are_not_silently_collapsed():
    with pytest.raises(ValueError):
        canonical_eye_layout(torch.zeros(2, 2, 721), 2, 721)
