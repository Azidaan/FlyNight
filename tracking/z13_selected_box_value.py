"""Fixed ridge value of one target-conditioned complete-box response."""
import numpy as np


def fit_value(x,y):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite known training rows required')
    mean=x.mean(0);sd=x.std(0);sd=np.maximum(sd,1e-6)
    z=(x-mean)/sd;ym=float(y.mean());zc=z-z.mean(0)
    w=np.linalg.solve(zc.T@zc/len(z)+np.eye(x.shape[1]),zc.T@(y-ym)/len(z))
    return dict(mean=mean.tolist(),sd=sd.tolist(),weight=w.tolist(),bias=ym)


def predict_value(model,x):
    x=np.asarray(x,np.float64)
    return ((x-model['mean'])/model['sd'])@np.asarray(model['weight'])+model['bias']
