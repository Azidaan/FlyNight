"""Linear triangle interpolation that exactly reproduces constant channels.

The third vertex is an anchor. The two barycentric coordinates multiply only
vertex differences, avoiding roundoff variation from summing weighted constants.
No contrast cutoff, fitted tolerance, identity or annotation is involved.
"""
from __future__ import annotations
import numpy as np


def sample_linear(mesh,values,query):
    values=np.asarray(values,dtype=np.float64);query=np.asarray(query,dtype=np.float64)
    if values.ndim!=2 or len(values)!=len(mesh.points) or query.ndim<1 or query.shape[-1]!=2 or not np.isfinite(values).all() or not np.isfinite(query).all():
        raise ValueError('Finite nodes-by-channels values and two-dimensional queries required')
    si=mesh.find_simplex(query)
    if np.any(si<0):raise ValueError('Query outside the eye hull')
    transform=mesh.transform[si];weights=np.einsum('...ij,...j->...i',transform[...,:2,:],query-transform[...,2,:]);vertices=values[mesh.simplices[si]];anchor=vertices[...,2,:]
    return anchor+np.einsum('...v,...vc->...c',weights,vertices[...,:2,:]-anchor[...,None,:])
