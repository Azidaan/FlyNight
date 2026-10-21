"""Guarded local quadratic peak vs the existing continuous weighted center.

The old center is already continuous. This changes the local estimator only;
it neither selects a different response peak nor claims to solve distractors.
"""
import numpy as np
from scripts.z13_target_model import center_and_presence


def quadratic_peak(score,geom):
    score=np.asarray(score,dtype=np.float64)
    if not np.isfinite(score).all():raise ValueError('Nonfinite response')
    baseline,presence=center_and_presence(score,geom)
    k=int(np.argmax(score));nb=np.unique(geom.neighbors[k])
    if len(nb)<6:return baseline,presence,'insufficient_support'
    xy=(geom.points[nb]-geom.points[k])/geom.spacing;x,y=xy.T
    A=np.stack([x*x,2*x*y,y*y,x,y,np.ones(len(x))],1)
    z=(score[nb]-score[k])/(float(score.std())+1e-9)
    coeff,_,rank,_=np.linalg.lstsq(A,z,rcond=None)
    if rank<6:return baseline,presence,'rank_deficient'
    a,b,c,d,e,_=coeff;H=2*np.array([[a,b],[b,c]])
    if np.linalg.eigvalsh(H).max()>=-1e-8 or np.linalg.cond(H)>1e6:return baseline,presence,'not_stable_concave'
    delta=-np.linalg.solve(H,[d,e]);center=geom.points[k]+geom.spacing*delta
    # Keep the solution inside this discrete peak's Voronoi cell. This is a
    # geometric trust region, not a threshold fitted to evaluation outcomes.
    if int(np.argmin(np.linalg.norm(geom.points-center,axis=1)))!=k:return baseline,presence,'outside_cell'
    return center,presence,'quadratic'
