"""Тот же FlyVis с детерминированной GPU-редукцией, без ослабления сравнений."""
from functools import wraps
from tracking.b1_tracker_runtime import ResidualEngine as OriginalEngine


def deterministic_call(function):
    @wraps(function)
    def execute(*args, **kwargs):
        import torch
        original = torch.are_deterministic_algorithms_enabled()
        original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return function(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(original, warn_only=original_warn_only)
    return execute


class DeterministicResidualEngine(OriginalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.arm in ('C', 'D'):
            self.bb.init = deterministic_call(self.bb.init)
            self.bb.step = deterministic_call(self.bb.step)
