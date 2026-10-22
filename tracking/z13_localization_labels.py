"""One intervention: target proximity rather than gain over Kalman in labels."""
import numpy as np


def localization_targets(labels, centers, mask):
    centers = np.asarray(centers)
    mask = np.asarray(mask, dtype=bool)
    if centers.shape != (len(labels), 3, 2) or mask.shape != (len(labels), 3):
        raise ValueError('Expected aligned three-candidate rows')
    out = np.zeros(mask.shape, dtype=np.float32)
    for i, row in enumerate(labels):
        if row['kind'] == 1:
            if row['scale'] <= 0:
                raise ValueError('Nonpositive target scale')
            distance = np.linalg.norm(centers[i] - np.asarray(row['gt']), axis=1)
            out[i] = (distance <= .25 * row['scale']) & mask[i]
    return out
