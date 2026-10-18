"""Differentiable implementation of the frozen per-target ridge readout.

Inputs are observed exemplar features/targets and search features only.
Search annotations are never inputs to this readout.
"""


def ridge_scores_torch(x_ex, target, x_search, lambdas):
    import torch
    if x_ex.ndim != 2 or target.shape != (len(x_ex),) or x_search.shape[-1] != x_ex.shape[-1]:
        raise ValueError('Invalid exemplar/target/search dimensions')
    if any(float(lam) <= 0 for lam in lambdas):
        raise ValueError('Positive regularization required')
    if not all(bool(torch.isfinite(a).all()) for a in (x_ex, target, x_search)):
        raise ValueError('Nonfinite readout input')
    mu = x_ex.mean(0)
    xc = x_ex - mu
    yc = target - target.mean()
    g = xc.T @ xc / len(xc)
    b = xc.T @ yc / len(xc)
    eye = torch.eye(len(g), dtype=g.dtype, device=g.device)
    return torch.stack([(x_search - mu) @ torch.linalg.solve(g + float(lam) * eye, b)
                        + target.mean() for lam in lambdas])
