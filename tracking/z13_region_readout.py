"""Read a target-response field at detection regions instead of its single peak.

No annotations, track identities, classes or outcome labels are inputs here.
The surrounding tracker is held fixed by the replay harness.
"""
import numpy as np


def region_probabilities(score, points, boxes, supported):
    score = np.asarray(score, np.float64)
    points = np.asarray(points, np.float64)
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    supported = np.asarray(supported, bool)
    assert score.shape == (len(points),) and points.shape[1] == 2
    assert supported.shape == (len(boxes),) and np.isfinite(score).all()
    centered = (score-score.mean()) / (score.std()+1e-9)
    logits = np.full(len(boxes), -np.inf)
    valid = supported.copy()
    for j, box in enumerate(boxes):
        mask = ((points[:,0] >= box[0]) & (points[:,0] <= box[2]) &
                (points[:,1] >= box[1]) & (points[:,1] <= box[3]))
        valid[j] &= bool(mask.any())
        if valid[j]:
            logits[j] = float(centered[mask].mean())
    # NULL logit is fixed at zero; an unsupported region gets zero probability.
    offset = max(0., float(logits.max(initial=-np.inf)))
    weights = np.exp(logits-offset)
    null = np.exp(-offset)
    return weights / (null+weights.sum()), valid


def cost_row(original, probabilities, confidences, supported, alpha):
    original = np.asarray(original, np.float64)
    p = np.asarray(probabilities, np.float64)
    conf = np.asarray(confidences, np.float64)
    supported = np.asarray(supported, bool)
    assert original.shape == p.shape == conf.shape == supported.shape
    assert 0 <= alpha <= 1 and np.isfinite(original).all()
    if alpha == 0:
        return original.copy()
    result = original.copy()
    result[supported] = ((1-alpha)*original[supported] +
                         alpha*(1-conf[supported]*p[supported]))
    return result
