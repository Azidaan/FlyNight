"""Per-target readout from separate pre-loss observations. No evaluation labels.

The same feature history reaches the last-observation control and the memory
readout. Memory changes the examples used to fit the target model, not the core.
"""
from __future__ import annotations
import numpy as np


def observation_slots(frame_numbers):
    """Use all distinct observations after fade-in: exemplar positions 1,2,3.

Position zero initializes the core; it is not a regular observation step.
If the short observed history was padded, keep the last occurrence per frame.
"""
    f=np.asarray(frame_numbers)
    if f.shape!=(4,) or np.any(np.diff(f)<0):
        raise ValueError('Four chronological pre-loss frame numbers required')
    latest={int(f[t]):t for t in (1,2,3)}
    return sorted(latest.values())


def memory_scores(features, masks, points, spacing, search, lambdas, frames):
    """Fit one ridge model from a bank of distinct observed feature/target maps.

features: (4,N,D), masks: (4,N); search: (S,N,D).  Only observed boxes enter
the Gaussian targets. Query GT, future frames and queried labels are absent.
"""
    from tracking.z13_target_center import observed_center_target,ridge_scores
    features=np.asarray(features,dtype=np.float64)
    masks=np.asarray(masks,dtype=bool)
    if features.ndim!=3 or features.shape[:2]!=masks.shape or len(features)!=4:
        raise ValueError('Mismatched observation bank')
    slots=observation_slots(frames)
    targets=[observed_center_target(masks[t],points,spacing) for t in slots]
    x=np.concatenate([features[t] for t in slots])
    return ridge_scores(x,np.concatenate(targets),search,lambdas)
