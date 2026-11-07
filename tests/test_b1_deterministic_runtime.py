import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from tracking.b1_deterministic_runtime import deterministic_call


def test_determinism_scope_preserves_external_setting_even_on_failure():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        @deterministic_call
        def success():
            assert torch.are_deterministic_algorithms_enabled()
            return 7
        assert success() == 7
        assert not torch.are_deterministic_algorithms_enabled()
        @deterministic_call
        def failure():
            assert torch.are_deterministic_algorithms_enabled()
            raise ValueError('test')
        import pytest
        with pytest.raises(ValueError): failure()
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
