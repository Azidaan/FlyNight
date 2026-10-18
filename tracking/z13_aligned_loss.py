"""Training-only query labels; inference readout remains annotation-free."""
import numpy as np


def query_target(observed_mask, points, spacing, true_center):
    points = np.asarray(points, dtype=float)
    mask = np.asarray(observed_mask, dtype=bool)
    if points.shape != (len(mask), 2) or not mask.any() or spacing <= 0:
        raise ValueError('Valid observed exemplar geometry required')
    if true_center is None:
        return np.zeros(len(points), dtype=float)
    extent = np.ptp(points[mask], axis=0) + spacing
    sigma = np.maximum(extent / 4, spacing / 2)
    return np.exp(-.5 * np.sum(((points - np.asarray(true_center)) / sigma) ** 2, axis=1))


def query_loss(scores, target):
    """DiMP robust squared query loss; fixed foreground label threshold0.05."""
    import torch
    residual = torch.where(target > .05, scores - target, scores.relu())
    return residual.square().mean()
