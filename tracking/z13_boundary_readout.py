"""Optional response abstention at the outer hexagonal ring; no tracker inputs."""
import numpy as np


def outer_ring(geometry):
    n=len(geometry.points)
    return np.asarray([len(set(map(int,nb))-{i})<6 for i,nb in enumerate(geometry.neighbors)],dtype=bool)


def read_response(score,geometry,enabled=True):
    from tracking.z13_ridge_localizer import map_center
    values=np.asarray(score,np.float64)
    if values.shape!=(len(geometry.points),) or not np.isfinite(values).all():
        raise ValueError('One finite response per hexagonal column required')
    center,confidence=map_center(values,geometry)
    accepted=not (enabled and outer_ring(geometry)[int(np.argmax(values))])
    return center,confidence,bool(accepted)
