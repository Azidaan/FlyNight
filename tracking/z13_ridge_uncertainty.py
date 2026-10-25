"""Template-only conditional ridge uncertainty; no query outcome inputs.

The empirical Gaussian linear model is an approximation: correlated retinal
columns and misspecified features do not support calibrated coverage claims.
"""
import numpy as np


def ridge_posterior(x_ex,target,x_search,lam):
    x=np.asarray(x_ex,np.float64);y=np.asarray(target,np.float64);s=np.asarray(x_search,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or s.shape[-1]!=x.shape[1] or lam<=0 or not all(np.isfinite(v).all() for v in (x,y,s)):
        raise ValueError('Finite aligned exemplar, target and search features required')
    n,d=x.shape;mu=x.mean(0);xc=x-mu;yc=y-y.mean();g=xc.T@xc/n;a=g+lam*np.eye(d)
    w=np.linalg.solve(a,xc.T@yc/n);inverse=np.linalg.solve(a,np.eye(d))
    df=float(np.trace(g@inverse));noise=float(np.sum((xc@w-yc)**2)/max(n-df-1,1.))
    covariance=(noise/n)*(inverse+inverse.T)/2
    centered=s-mu;mean=centered@w+y.mean()
    variance=np.maximum(np.sum((centered@covariance)*centered,axis=-1),0.)
    return mean,np.sqrt(variance),dict(noise_variance=noise,effective_df=df,template_points=n)


def region_probabilities(score,points,boxes,supported,uncertainty=None,mode='old'):
    from tracking.z13_region_readout import region_probabilities as original
    if mode=='old':return original(score,points,boxes,supported)
    if mode not in ('posterior','constant'):raise ValueError('Unknown uncertainty control')
    score=np.asarray(score,np.float64);points=np.asarray(points,np.float64);boxes=np.asarray(boxes,np.float64).reshape(-1,4)
    sd=np.asarray(uncertainty,np.float64);valid=np.asarray(supported,bool).copy()
    if sd.shape!=score.shape or not np.isfinite(sd).all() or np.any(sd<0):raise ValueError('Aligned finite nonnegative uncertainty required')
    scale=score.std()+1e-9;z=(score-score.mean())/scale;u=sd/scale
    if mode=='constant':u=np.full_like(u,u.mean())
    logits=np.full(len(boxes),-np.inf)
    for j,b in enumerate(boxes):
        mask=((points>=b[:2])&(points<=b[2:])).all(1);valid[j]&=bool(mask.any())
        if valid[j]:logits[j]=float(z[mask].mean()-1.96*u[mask].mean())
    offset=max(0.,float(logits.max(initial=-np.inf)));weights=np.exp(logits-offset)
    return weights/(np.exp(-offset)+weights.sum()),valid
