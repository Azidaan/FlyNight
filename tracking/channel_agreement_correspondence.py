"""Experimental agreement at an already fixed spatial correspondence.

Lin's concordance retains mean/scale disagreement discarded by Pearson.
This is not an object-ID rule, a threshold, or an intervention in FlyVis.
"""
from __future__ import annotations
import numpy as np

def agreement_channels(reference,current,*,mean_penalty=True):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(current,dtype=np.float64)
    if a.ndim!=2 or a.shape!=b.shape or not len(a) or not a.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Equal finite cells-by-channels patches required')
    active_a=np.ptp(a,axis=0)!=0;active_b=np.ptp(b,axis=0)!=0
    if not active_a.any() or not active_b.any():return None
    ma=a.mean(0);mb=b.mean(0);x=a-ma;y=b-mb
    x[:,~active_a]=0;y[:,~active_b]=0
    va=(x*x).mean(0);vb=(y*y).mean(0);cov=(x*y).mean(0)
    den=va+vb+((ma-mb)**2 if mean_penalty else 0.)
    value=np.divide(2*cov,den,out=np.zeros_like(den),where=den>0)
    value[~active_a]=0
    return value*a.shape[1]/int(active_a.sum())
