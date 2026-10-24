"""Counterfactual target specificity at the original peak; no label inputs."""
import numpy as np


def specificity_gap(maps):
    maps=np.asarray(maps,np.float64)
    if maps.shape!=(4,721) or not np.isfinite(maps).all():
        raise ValueError('One own and three complete foreign response maps required')
    z=(maps-maps.mean(1,keepdims=True))/(maps.std(1,keepdims=True)+1e-9)
    peak=int(np.argmax(maps[0]))
    return float(z[0,peak]-z[1:,peak].mean())


def select(rows,key):
    """Frozen diagnostic1% coverage on the identical cohort; label breaks ties."""
    import math
    k=max(1,math.ceil(.01*len(rows)))
    return sorted(rows,key=lambda x:(-x[key],x['label']))[:k]
