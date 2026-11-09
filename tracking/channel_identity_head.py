"""Experimental causal identity decision from spatial channel correspondence.

This does not change FlyVis weights, detections or tracker state. Spatial search
uses the previous equal-channel score; labels never choose a shift or cells.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from tracking.channel_balanced_correspondence import _center_pair
from tracking.hex_linear_interpolation import sample_linear


def spatial_channel_features(mesh, reference, current, cells, query):
    """Return channel correlations at the fixed old reader's winning shift.

    The mean of the returned vector equals the equal-reference-channel score.
    Undefined evidence remains undefined. A caller must use its geometry-only
    fallback rather than treating a constant patch as positive evidence.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    idx = np.asarray(cells, dtype=np.int64)
    q = np.asarray(query, dtype=np.float64)
    if (a.ndim != 2 or b.shape != a.shape or len(a) != len(mesh.points)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or idx.ndim != 1 or len(idx) < 3 or len(np.unique(idx)) != len(idx)
            or (idx < 0).any() or (idx >= len(a)).any()
            or q.ndim != 3 or q.shape[0] != len(idx) or q.shape[2] != 2
            or not q.shape[1] or not np.isfinite(q).all()
            or (mesh.find_simplex(q) < 0).any()):
        raise ValueError('Finite aligned maps, unique supported cells and in-hull queries required')
    patches = sample_linear(mesh, b, q).transpose(1, 0, 2)
    correlations = []
    scores = []
    for patch in patches:
        x, y, an, bn, active_a, active_b = _center_pair(a[idx], patch)
        if not active_a.any() or not active_b.any():
            correlations.append(None)
            scores.append(None)
            continue
        den = an * bn
        rho = np.divide((x*y).sum(0), den, out=np.zeros_like(an), where=den > 0)
        # Preserve the old reference-active denominator across candidates.
        vector = rho * a.shape[1] / int(active_a.sum())
        correlations.append(vector)
        scores.append(float(rho.sum() / active_a.sum()))
    valid = [i for i, value in enumerate(scores) if value is not None]
    winner = max(valid, key=lambda i: scores[i]) if valid else None
    return dict(valid=winner is not None, winner=winner, scores=scores,
                channels=None if winner is None else correlations[winner],
                scalar=None if winner is None else scores[winner])


def causal_geometry(reference_box, current_box, original_cost, gap, cells):
    """Seven features available before assignment; no ID or label inputs."""
    a = np.asarray(reference_box, dtype=np.float64)
    b = np.asarray(current_box, dtype=np.float64)
    if (a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a,b,original_cost,gap,cells]).all()
            or min(*(a[2:]-a[:2]), *(b[2:]-b[:2])) <= 0 or gap < 0 or cells < 3):
        raise ValueError('Positive boxes, nonnegative gap and supported cells required')
    size = a[2:]-a[:2]
    ratio = (b[2:]-b[:2])/size
    displacement = ((b[:2]+b[2:])-(a[:2]+a[2:]))/(2*size)
    return np.r_[original_cost, gap, np.log(ratio), displacement, np.log1p(cells)]


def geometry_transform(train, test):
    """Fit geometry standardization only on the current training fold."""
    x = np.asarray(train, dtype=np.float64)
    y = np.asarray(test, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned training and test matrices required')
    mean = x.mean(0)
    scale = x.std(0)
    constant = np.ptp(x, axis=0) == 0
    scale[constant] = 1
    if (scale <= 0).any():
        raise ValueError('Unexpected nonpositive geometry scale')
    return (x-mean)/scale, (y-mean)/scale, dict(mean=mean, scale=scale)


def objective(theta, matrix, labels, sample_weights):
    """Mean class-balanced logistic loss plus ||w||^2/2; free intercept.

    The final coordinate of theta is the intercept. Its penalty is zero.
    Returns the exact value, gradient and Hessian for mathematical checks.
    """
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    sw = np.asarray(sample_weights, dtype=np.float64)
    if (x.ndim != 2 or theta.shape != (x.shape[1]+1,) or y.shape != (len(x),)
            or sw.shape != y.shape or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(theta).all() or not np.isfinite(sw).all()
            or not np.isin(y, [0,1]).all() or (sw <= 0).any()):
        raise ValueError('Finite binary classification data and positive sample weights required')
    design = np.column_stack([x, np.ones(len(x))])
    z = design @ theta
    probability = expit(z)
    penalty = theta.copy()
    penalty[-1] = 0
    loss = float(np.mean(sw * (np.logaddexp(0, z)-y*z)) + .5*(penalty@penalty))
    gradient = design.T @ (sw*(probability-y))/len(x) + penalty
    hessian = (design.T * (sw*probability*(1-probability))) @ design / len(x)
    hessian += np.diag(np.r_[np.ones(x.shape[1]), 0.])
    return loss, gradient, hessian


def fit_ridge_logistic(matrix, labels):
    """One fixed deterministic fit, without tuning or calibrated probabilities."""
    x = np.asarray(matrix, dtype=np.float64)
    raw_y = np.asarray(labels)
    if x.ndim != 2 or raw_y.shape != (len(x),) or set(raw_y.tolist()) != {0,1}:
        raise ValueError('Both classes required in every training fold')
    y = raw_y.astype(np.int64)
    counts = np.bincount(y, minlength=2)
    sw = len(y) / (2*counts[y])
    theta = np.zeros(x.shape[1]+1)
    for iteration in range(100):
        loss, gradient, hessian = objective(theta, x, y, sw)
        residual = float(np.max(np.abs(gradient)))
        if residual <= 1e-9:
            return dict(theta=theta, loss=loss, gradient_max=residual, iterations=iteration,
                        class_counts=counts, threshold=0., regularization=1.)
        direction = np.linalg.solve(hessian, gradient)
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope <= 0:
            raise ArithmeticError('Invalid Newton descent direction')
        step = 1.
        for _ in range(60):
            trial = theta-step*direction
            value = objective(trial, x, y, sw)[0]
            if value <= loss-1e-4*step*slope:
                theta = trial
                break
            step *= .5
        else:
            # Near machine precision a strict Armijo comparison may stall.
            if residual <= 1e-7:
                return dict(theta=theta, loss=loss, gradient_max=residual, iterations=iteration,
                            class_counts=counts, threshold=0., regularization=1.)
            raise ArithmeticError('Newton line search did not converge')
    raise ArithmeticError('Fixed Newton iteration limit reached')


def decision_function(model, matrix):
    x = np.asarray(matrix, dtype=np.float64)
    theta = np.asarray(model['theta'], dtype=np.float64)
    if x.ndim != 2 or theta.shape != (x.shape[1]+1,) or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix matching fitted weights required')
    return x @ theta[:-1] + theta[-1]
