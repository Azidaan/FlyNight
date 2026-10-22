"""Observed-only target/background weighting of the fixed Gaussian ridge target."""
import numpy as np


def spatial_weights(mask, balanced=True):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=1 or not mask.any():raise ValueError('Nonempty observed target required')
    n=len(mask);weights=np.ones(n,dtype=np.float64)/n
    if balanced and not mask.all():
        weights[mask]=.5/mask.sum();weights[~mask]=.5/(~mask).sum()
    return weights


def fit_observed(features,target,mask,lam,balanced=True):
    x=np.asarray(features,np.float64);y=np.asarray(target,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or np.asarray(mask).shape!=y.shape or lam<=0 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid observed ridge input')
    q=spatial_weights(mask,balanced);mu=q@x;bias=float(q@y);xc=x-mu;yc=y-bias
    weight=np.linalg.solve(xc.T@(q[:,None]*xc)+lam*np.eye(x.shape[1]),xc.T@(q*yc))
    return mu,weight,bias
