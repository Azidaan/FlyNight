from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/neural-tensor-interventions/scripts'))
from intervention import scale_nodes

def test_identity_and_selected_only_without_mutating_source():
    src = torch.tensor([[[1., -2., 3., 4.]]], dtype=torch.float64)
    before = src.clone()
    assert torch.equal(scale_nodes(src, [1, 3], 1.), src)
    out = scale_nodes(src, [1, 3], .9)
    assert torch.equal(src, before)
    assert torch.equal(out[..., [0, 2]], src[..., [0, 2]])
    assert torch.equal(out[..., [1, 3]], src[..., [1, 3]] * .9)
    assert out.dtype == src.dtype and out.device == src.device

@pytest.mark.parametrize('indices,factor,error', [([], .9, ValueError), ([1, 1], .9, ValueError), ([-1], .9, IndexError), ([4], .9, IndexError), ([1], 1.1, ValueError)])
def test_invalid_interventions_rejected(indices, factor, error):
    with pytest.raises(error):
        scale_nodes(torch.ones(4), indices, factor)
