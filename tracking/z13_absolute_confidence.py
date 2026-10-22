"""Raw response maximum as a score; unchanged center, no probability claim."""
import numpy as np


def read_response(score,geometry,enabled=True):
    from tracking.z13_ridge_localizer import map_center
    values=np.asarray(score,np.float64)
    if values.shape!=(len(geometry.points),) or not np.isfinite(values).all():raise ValueError('Finite map required')
    center,z=map_center(values,geometry)
    return center,float(values.max()) if enabled else z,True
