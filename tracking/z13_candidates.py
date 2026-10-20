"""Local response candidates. Pure feature readout; no search labels or GT API."""
from __future__ import annotations
import numpy as np


def candidates(score, features, geom, k=3):
    score=np.asarray(score,dtype=np.float64)
    features=np.asarray(features,dtype=np.float64)
    if score.shape!=(len(geom.points),) or features.shape[0]!=len(score):
        raise ValueError('Candidate feature/geometry shape mismatch')
    if not np.isfinite(score).all() or not np.isfinite(features).all():
        raise ValueError('Nonfinite response/features')
    if k<1:raise ValueError('Positive candidate count required')
    # Rank1 is exactly the existing argmax. Subsequent candidates must be
    # local maxima, separated by two lattice spacings from every kept peak.
    order=np.argsort(-score,kind='stable');peaks=[int(np.argmax(score))]
    for i in order:
        if len(peaks)>=k:break
        if score[i]<score[geom.neighbors[i]].max():continue
        if min(np.linalg.norm(geom.points[i]-geom.points[j]) for j in peaks)<2*geom.spacing:continue
        peaks.append(int(i))
    centers=[];desc=[];heights=[];sd=float(score.std())+1e-9
    for i in peaks:
        nb=np.unique(geom.neighbors[i]);s=score[nb]
        w=np.exp((s-s.max())/sd);w/=w.sum()
        centers.append(w@geom.points[nb]);desc.append(w@features[nb])
        heights.append((score[i]-score.mean())/sd)
    return dict(indices=np.asarray(peaks,np.int32),centers=np.asarray(centers),
                features=np.asarray(desc),scores=np.asarray(heights))
