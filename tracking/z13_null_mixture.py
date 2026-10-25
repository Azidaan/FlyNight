"""Alternative readout contract: NULL leaves the surrounding cost row unchanged.

This is a new application of the FlyVis readout, not a service correction of
the former 'NULL means no candidate' model or an intrinsic core improvement.
"""
import numpy as np


def cost_row(original,probabilities,confidences,supported,alpha,mode='old'):
    from tracking.z13_region_readout import cost_row as previous
    if mode=='old':return previous(original,probabilities,confidences,supported,alpha)
    if mode!='null_mix':raise ValueError('Unknown NULL contract')
    old=np.asarray(original,np.float64);p=np.asarray(probabilities,np.float64);conf=np.asarray(confidences,np.float64);valid=np.asarray(supported,bool)
    if not (old.shape==p.shape==conf.shape==valid.shape) or not all(np.isfinite(x).all() for x in (old,p,conf)) or np.any(p<0) or np.any(p[~valid]!=0) or not 0<=alpha<=1:
        raise ValueError('Aligned finite probabilities/confidence/support required')
    q=float(p.sum());assert q<=1+1e-12;q=min(q,1.)
    if alpha==0 or q==0:return old.copy()
    result=old.copy();result[valid]=old[valid]+alpha*(q*(1-old[valid])-conf[valid]*p[valid])
    return result
