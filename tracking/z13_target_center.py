"""Per-target ridge readout: binary region control and observed-center Gaussian target.

Only observed exemplar mask/geometry and feature arrays enter this API.
No search labels, true centers or future observations are accepted.
"""
import numpy as np


def observed_center_target(mask, points, spacing):
    mask=np.asarray(mask,dtype=bool); points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape!=(len(mask),2) or not mask.any() or spacing<=0:
        raise ValueError('Nonempty observed mask and positive lattice spacing required')
    selected=points[mask]
    lo,hi=selected.min(0),selected.max(0)
    center=(lo+hi)/2
    # Reconstruct occupied extent on the same lattice. No current Kalman or GT box.
    extent=hi-lo+spacing
    sigma=np.maximum(extent/4,spacing/2)
    return np.exp(-.5*np.sum(((points-center)/sigma)**2,axis=1))


def ridge_scores(x_ex, target, x_search, lambdas):
    x_ex=np.asarray(x_ex,dtype=np.float64); target=np.asarray(target,dtype=np.float64)
    x_search=np.asarray(x_search,dtype=np.float64)
    if target.shape!=(len(x_ex),) or not all(np.isfinite(a).all() for a in (x_ex,target,x_search)):
        raise ValueError('Invalid features/target')
    if any(lam<=0 for lam in lambdas): raise ValueError('Regularization must be positive')
    mu=x_ex.mean(0); xc=x_ex-mu; yc=target-target.mean()
    g=xc.T@xc/len(xc); b=xc.T@yc/len(xc)
    return np.stack([(x_search-mu)@np.linalg.solve(g+lam*np.eye(len(g)),b)+target.mean() for lam in lambdas])
