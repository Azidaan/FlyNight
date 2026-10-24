"""Target-versus-observed-background readout with a fixed null score of zero.

Only previously observed descriptors and current candidate descriptors enter
this API. No labels, future frames, tracker errors or query masks are accepted.
This is a small fixed prototype inspired by DaSiamRPN's contrastive inference;
it does not implement its learned embedding, incremental updates or tracker.
"""
import numpy as np


def unit_rows(values):
    a=np.asarray(values,np.float64)
    if a.ndim!=2 or a.shape[1]!=68 or not np.isfinite(a).all():
        raise ValueError('Finite 68-dimensional descriptors required')
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)


def contrast_scores(candidate_features, observed_target, observed_background):
    """One fixed contrast: cos(target,query)-mean cos(background,query)."""
    query=unit_rows(candidate_features)
    target=unit_rows(np.asarray(observed_target).reshape(1,68))[0]
    background=unit_rows(np.asarray(observed_background).reshape(-1,68))
    positive=query@target
    negative=query@background.mean(0) if len(background) else np.zeros(len(query))
    return positive,positive-negative


def background_indices(centers_eye, observed_mask, geometry):
    """A complete local neighborhood outside the last observed target mask."""
    mask=np.asarray(observed_mask,bool)
    if mask.shape!=(len(geometry.points),) or not mask.any():
        raise ValueError('Nonempty observed target mask required')
    answer=[]
    for j,center in enumerate(np.asarray(centers_eye,float)):
        k=geometry.nearest(center)
        if not mask[np.unique(geometry.neighbors[k])].any():answer.append(j)
    return answer


def quota_decisions(scores, fraction=.01):
    """Preserved fixed quota; a nonpositive target-background margin abstains."""
    x=np.asarray(scores,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or fraction!=.01:
        raise ValueError('Only fixed one-percent prototype quota is supported')
    n=max(1,int(np.ceil(len(x)*fraction)))
    order=np.argsort(-x,kind='stable')[:n]
    selected=np.zeros(len(x),bool);selected[order]=x[order]>0
    return selected
