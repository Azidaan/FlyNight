"""Causal spatial patch filter trained against real reference-frame background.

Transfers the real-negative-patch principle of BACF (ICCV 2017, Eq. 3).
This is a bounded direct ridge adaptation, not the published FFT/ADMM tracker.
No labels, current feature map, IDs or future frames enter fit().
"""
from __future__ import annotations
import numpy as np


def unit_patch(patch):
    a=np.asarray(patch,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Finite cells-by-channels required')
    active=np.ptp(a,axis=0)!=0
    a=a-a.mean(0)
    a[:,~active]=0
    norm=np.linalg.norm(a)
    return None if norm==0 else a.ravel()/norm


class BackgroundPatchFilter:
    """One target term, mean negative loss, fixed unit ridge regularization."""
    def fit(self,reference,negative_patches):
        self.weights=None
        positive=unit_patch(reference)
        if positive is None:return False
        negative_patches=list(negative_patches)
        if any(np.shape(p)!=np.shape(reference) for p in negative_patches):raise ValueError('Fixed negative spatial support required')
        negatives=[unit_patch(p) for p in negative_patches]
        negatives=[p for p in negatives if p is not None]
        X=np.stack([positive]+[p/np.sqrt(len(negatives)) for p in negatives])
        y=np.zeros(len(X));y[0]=1.
        coefficients=np.linalg.solve(X@X.T+np.eye(len(X)),y)
        self.weights=X.T@coefficients
        self.normalizer=float(positive@self.weights)
        # ||positive||=1 and tr(X'X)<=2 imply a stable positive self response.
        assert self.normalizer>=1/3-1e-12
        self.shape=np.shape(reference);self.negative_count=len(negatives)
        return True

    def score(self,patch):
        if getattr(self,'weights',None) is None:raise RuntimeError('A nondegenerate reference must be fitted first')
        if np.shape(patch)!=self.shape:raise ValueError('Fixed spatial support required')
        z=unit_patch(patch)
        return None if z is None else float(z@self.weights/self.normalizer)


def background_offsets(points,indices,box,mesh,limit=64):
    """Real complete patches disjoint from the observed reference box, no wrap."""
    points=np.asarray(points,float);indices=np.asarray(indices,int);box=np.asarray(box,float)
    if limit<1 or box.shape!=(4,) or min(box[2:]-box[:2])<=0:raise ValueError('Positive box and background count required')
    offsets=points-(box[:2]+box[2:])/2
    shifted=box[None,:]+np.tile(offsets,(1,2))
    overlap=np.maximum(0,np.minimum(shifted[:,2:],box[2:])-np.maximum(shifted[:,:2],box[:2])).prod(1)
    queries=points[indices][None]+offsets[:,None]
    keep=(overlap==0)&(mesh.find_simplex(queries)>=0).all(1)
    ids=np.flatnonzero(keep)
    # Geometry-only farthest-point coverage, independent of activity and labels.
    if len(ids)>limit:
        chosen=[int(ids[0])]
        while len(chosen)<limit:
            distance=((points[ids,None]-points[np.array(chosen)][None])**2).sum(-1).min(1)
            distance[np.isin(ids,chosen)]=-1
            chosen.append(int(ids[np.argmax(distance)]))
        ids=np.asarray(chosen)
    return offsets[ids]
