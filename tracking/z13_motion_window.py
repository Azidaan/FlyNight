"""Spatial window inside map localization; no annotations or future inputs."""
import numpy as np


def motion_window_scores(scores,points,prediction,eye_size,influence):
    s=np.asarray(scores,np.float64);xy=np.asarray(points,np.float64);p=np.asarray(prediction,np.float64)
    if s.ndim!=1 or not len(s) or xy.shape!=(len(s),2) or p.shape!=(2,) or eye_size<=0 or not 0<=influence<=1:
        raise ValueError('Invalid map/window geometry')
    if not all(np.isfinite(x).all() for x in (s,xy,p)):raise ValueError('Finite map/window inputs required')
    if influence==0:return s.copy()
    response=s-s.min();total=response.sum();response=response/total if total>0 else np.full_like(s,1/len(s))
    delta=np.clip((xy-p)/eye_size,-.5,.5)
    window=np.prod(.5*(1+np.cos(2*np.pi*delta)),axis=1);den=window.sum()
    window=window/den if den>0 else response.copy()
    return (1-influence)*response+influence*window


def motion_window_center(scores,geom,prediction,eye_size,influence):
    """Change selected column only; retain original local weighting/confidence."""
    from scripts.z13_target_model import center_and_presence
    if influence==0:return center_and_presence(scores,geom)
    selected=int(np.argmax(motion_window_scores(scores,geom.points,prediction,eye_size,influence)))
    s=np.asarray(scores,np.float64);nb=np.unique(geom.neighbors[selected]);sd=float(s.std())+1e-9
    w=np.exp((s[nb]-s[nb].max())/sd)
    center=(geom.points[nb]*w[:,None]).sum(0)/w.sum()
    return center,float((s.max()-s.mean())/sd)
