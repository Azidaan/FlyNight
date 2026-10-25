"""Uncompressed target/activity interaction within the regional FlyVis readout."""
import numpy as np


def product_scale(x, seed, mask):
    """Fit an RMS per original channel from observable fit84 regions only."""
    x=np.asarray(x,np.float64);seed=np.asarray(seed,np.float64);mask=np.asarray(mask,bool)
    if x.ndim!=3 or x.shape[-1]!=68 or seed.shape!=(len(x),68) or mask.shape!=x.shape[:2] or not mask.any():
        raise ValueError('Aligned nonempty original activity, target and region mask required')
    value=(x*seed[:,None,:])[mask]
    if not np.isfinite(value).all():raise ValueError('Finite observable interactions required')
    return np.maximum(np.sqrt(np.mean(value*value,axis=0)),1e-6).astype(np.float32)


def product_features(x, seed, scale):
    x=np.asarray(x,np.float32);seed=np.asarray(seed,np.float32);scale=np.asarray(scale,np.float32)
    if x.ndim!=3 or x.shape[-1]!=68 or seed.shape!=(len(x),68) or scale.shape!=(68,) or not np.all(scale>0):
        raise ValueError('Aligned original channels and positive fit-only scale required')
    value=x*seed[:,None,:]/scale
    if not np.isfinite(value).all():raise ValueError('Finite interaction required')
    return value


def make_product_ranker(torch, activity=True):
    from tracking.z13_region_assignment_context import make_context_ranker
    model=make_context_ranker(torch,activity=activity)
    model.score[0]=torch.nn.Linear(162,32)
    return model
