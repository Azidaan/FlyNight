"""Action-space bounds and explicitly conditional sensitivity, never model forecasts."""
import numpy as np


def conditional_net(reserve, recovery, false_harm):
    values = np.asarray(reserve, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError('Reserve must be a fraction')
    if not 0 <= recovery <= 1 or not 0 <= false_harm <= 1:
        raise ValueError('Conditional rates must be fractions')
    return recovery * values - false_harm * (1 - values)


def required_recovery(reserve, target=.02, false_harm=.01):
    if not 0 < reserve <= 1:
        raise ValueError('Positive reserve required')
    return (target + false_harm * (1 - reserve)) / reserve


def helpful_box_mask(choices, good):
    if len(choices) != len(good):
        raise ValueError('Action binding mismatch')
    return any(int(choice) >= 0 and bool(value) for choice, value in zip(choices, good))
