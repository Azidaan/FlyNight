"""Construct the tested synthetic reference history from already observed inputs.

This is an optional template proposal, not a change of neural parameters or a
claim that the recurrent model reached equilibrium. The current history stays
untouched; callers still need the correspondence and ownership guards.
"""
import numpy as np


def terminal_reference_history(observed_views, extra_steps=30):
    """Repeat the last prepared input for original history length plus 30 steps."""
    views=np.asarray(observed_views)
    if views.ndim<2 or len(views)==0 or not np.isfinite(views).all():
        raise ValueError('A nonempty finite history of prepared inputs is required')
    if type(extra_steps) is not int or extra_steps<0:
        raise ValueError('A nonnegative integer step count is required')
    return np.repeat(views[-1:],len(views)+extra_steps,axis=0).copy()
