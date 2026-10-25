"""Add two local spatial derivatives without discarding the old 68 features."""
import numpy as np

def derivative_projector(points,neighbors,spacing):
    points=np.asarray(points,dtype=np.float64);neighbors=np.asarray(neighbors)
    if neighbors.shape!=(len(points),7) or spacing<=0:raise ValueError('Hexagonal geometry required')
    offsets=(points[neighbors[:,1:]]-points[:,None,:])/spacing
    if not np.all(np.linalg.matrix_rank(offsets)==2):raise ValueError('Two observable spatial directions required')
    return np.linalg.pinv(offsets)

def spatial_descriptor(raw,mean,sd,neighbors,projector):
    raw=np.asarray(raw,dtype=np.float64);mean=np.asarray(mean);sd=np.asarray(sd)
    dim=raw.shape[0]//2
    if raw.shape[0]!=2*dim or mean.shape!=(2*dim,) or sd.shape!=(2*dim,) or np.any(sd<=0):raise ValueError('Original channel and neighbor-mean normalization required')
    values=raw[:dim];differences=values[:,neighbors[:,1:]]-values[:,:,None]
    slopes=np.einsum('nqk,dnk->ndq',projector,differences)
    old=(raw.T-mean)/sd
    out=np.concatenate((old,slopes[:,:,0]/sd[:dim],slopes[:,:,1]/sd[:dim]),axis=1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite spatial descriptor')
    return out

def zero_extension(x):
    return np.concatenate((x,np.zeros_like(x)),axis=-1)
