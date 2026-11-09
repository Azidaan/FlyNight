"""Experimental contrast of an alignment response over fixed hexagonal shifts."""
import numpy as np
from tracking.channel_balanced_correspondence import _center_pair
from tracking.hex_linear_interpolation import sample_linear

AXES=np.asarray([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2],int)

def nonneighbors(axes=AXES):
    a=np.asarray(axes)
    if a.shape!=(19,2) or len(np.unique(a,axis=0))!=19 or not np.array_equal(a,a.astype(int)):
        raise ValueError('The19uniqueintegeraxialoffsets are required')
    if set(map(tuple,a))!=set(map(tuple,AXES)):raise ValueError('Same fixed radius2shift set required')
    d=a[:,None,:]-a[None,:,:];distance=np.maximum(np.abs(d).max(-1),np.abs(d.sum(-1)))
    return [np.flatnonzero(row>=2) for row in distance]

def subtract_sidelobes(profiles,winner,axes=AXES):
    r=np.asarray(profiles,float)
    if r.ndim!=2 or r.shape[0]!=19 or not np.isfinite(r).all() or not 0<=winner<19:raise ValueError('Finite19bychannels profile and winner required')
    u=nonneighbors(axes)[winner]
    return r[winner]-r[u].mean(0)

def recover_missing_profiles(mesh,reference,current,cells,query,ready):
    """Reuse the ready winning vector and scalar profile; derive only18missing vectors."""
    a=np.asarray(reference,float);b=np.asarray(current,float);idx=np.asarray(cells,int);q=np.asarray(query,float)
    if not ready['valid'] or len(ready['scores'])!=19 or any(x is None for x in ready['scores']):raise ValueError('All19readyshifts must be defined; do not drop samples or impute zeros')
    winner=int(ready['winner']);r=np.empty((19,a.shape[1]),float);r[winner]=ready['channels'];missing=np.array([j for j in range(19) if j!=winner])
    patches=sample_linear(mesh,b,q[:,missing,:]).transpose(1,0,2)
    for j,patch in zip(missing,patches):
        x,y,an,bn,active_a,active_b=_center_pair(a[idx],patch)
        if not active_a.any() or not active_b.any():raise ValueError('Undefinedshift; preserve cohort and stop readiness')
        rho=np.divide((x*y).sum(0),an*bn,out=np.zeros_like(an),where=an*bn>0)
        r[j]=rho*a.shape[1]/int(active_a.sum())
    error=float(abs(r.mean(1)-np.array(ready['scores'])).max())
    if error>1e-12:raise ArithmeticError(('Ready scalar profile mismatch',error))
    if winner!=max(range(19),key=lambda j:ready['scores'][j]):raise ArithmeticError('Changed old shift selection')
    return dict(profiles=r,contrast=subtract_sidelobes(r,winner),winner=winner,parity_error=error,ready_winner_reused=True,new_channel_vectors=18)
