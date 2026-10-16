"""Experimental spatial target for the Z13 readout; does not alter inference."""
from __future__ import annotations


def spatial_targets(torch, kinds, centers_eye, points_eye, sigma):
    """Normalized Gaussian on the eye grid; known absence has its own class.

    Unknown rows carry zero weight, never the absence label. Sigma is a fixed
    grid-scale hypothesis, not an estimate of annotation uncertainty.
    """
    if sigma <= 0:
        raise ValueError('Positive fixed sigma required')
    kinds = torch.as_tensor(kinds, device=points_eye.device, dtype=torch.long)
    centers_eye = torch.as_tensor(centers_eye, device=points_eye.device, dtype=points_eye.dtype)
    if kinds.ndim != 1 or centers_eye.shape != (len(kinds), 2):
        raise ValueError('One kind and xy center per row required')
    if not bool(((kinds == -1) | (kinds == 0) | (kinds == 1)).all()):
        raise ValueError('Unknown kind')
    result = points_eye.new_zeros((len(kinds), len(points_eye) + 1))
    positive = kinds == 1
    if bool(positive.any()):
        if not bool(torch.isfinite(centers_eye[positive]).all()):
            raise ValueError('Invalid positive center')
        distance = (points_eye[None] - centers_eye[positive, None]).square().sum(-1)
        result[positive, :-1] = (-distance / (2 * sigma * sigma)).softmax(-1)
    result[kinds == 0, -1] = 1
    return result, kinds >= 0


def distribution_loss(torch, logits, target, known):
    if logits.shape != target.shape:
        raise ValueError('Target/logit mismatch')
    if not bool(known.any()):
        return logits.sum() * 0
    return -(target[known] * logits[known].log_softmax(-1)).sum(-1).mean()
