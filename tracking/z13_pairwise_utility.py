"""A fixed pairwise term added to the existing linear utility model."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


def pair_indices(targets, limit=4096, seed=0):
    targets = np.asarray(targets)
    positive, negative = np.flatnonzero(targets == 1), np.flatnonzero(targets == 0)
    if not len(positive) or not len(negative):
        raise ValueError('Both known outcome classes are required')
    total = len(positive) * len(negative)
    rng = np.random.default_rng(seed)
    index = rng.choice(total, min(limit, total), replace=False)
    return positive[index // len(negative)], negative[index % len(negative)]


def loss_gradient(theta, x, y, difference, strength):
    w, bias = theta[:-1], theta[-1]
    z = x @ w + bias
    n = len(y)
    loss = np.mean(np.logaddexp(0, z) - y * z) + .5 * np.dot(w, w) / n
    residual = expit(z) - y
    gradient = np.r_[x.T @ residual / n + w / n, residual.mean()]
    if strength:
        margin = difference @ w
        loss += strength * np.logaddexp(0, -margin).mean()
        gradient[:-1] -= strength * (difference.T @ expit(-margin)) / len(margin)
    return float(loss), gradient


def leave_video_out(x, y, videos, strength, on_fold=None, check=None):
    """Fit scaler and objective on known outcomes from other videos only."""
    x, y, videos = np.asarray(x, float), np.asarray(y, int), np.asarray(videos)
    if strength not in (0., 1.) or not np.isfinite(x).all():
        raise ValueError('Only the two preregistered fixed objectives are supported')
    scores = np.full(len(y), np.nan)
    records = []
    unique = sorted(set(videos))
    for number, video in enumerate(unique, 1):
        if check:
            check()
        test = videos == video
        fit = (~test) & (y >= 0)
        scaler = StandardScaler().fit(x[fit])
        train = scaler.transform(x[fit])
        targets = y[fit]
        pos, neg = pair_indices(targets)
        difference = train[pos] - train[neg]
        result = minimize(loss_gradient, np.zeros(x.shape[1] + 1),
                          args=(train, targets, difference, strength), jac=True,
                          method='L-BFGS-B', callback=(lambda _: check()) if check else None,
                          options=dict(maxiter=500, gtol=1e-8, ftol=1e-12))
        if not result.success:
            raise RuntimeError('Utility optimizer failed: ' + str(result.message))
        scores[test] = expit(scaler.transform(x[test]) @ result.x[:-1] + result.x[-1])
        fit_videos = sorted(str(v) for v in set(videos[fit]))
        assert video not in fit_videos
        records.append(dict(test_video=str(video), fit_videos=fit_videos,
                            fit_known=int(fit.sum()), test_rows=int(test.sum()),
                            positive=int((targets == 1).sum()), pairs=len(pos),
                            iterations=int(result.nit), loss=float(result.fun),
                            coefficients=result.x.tolist(), mean=scaler.mean_.tolist(),
                            scale=scaler.scale_.tolist(), strength=strength))
        if on_fold:
            on_fold(number, len(unique), str(video))
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete held-out predictions')
    return scores, records
