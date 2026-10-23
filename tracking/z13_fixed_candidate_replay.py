"""Verify observed features at original immutable anchors, without peak selection."""
import numpy as np


def replay(score,features,geom,centers,scores,crop,eye_size):
    score=np.asarray(score,np.float64);features=np.asarray(features,np.float64);centers=np.asarray(centers,np.float64);scores=np.asarray(scores,np.float64)
    if not 0<len(centers)<=3 or len(scores)!=len(centers):raise ValueError('Original finite candidate count required')
    nb=np.asarray(geom.neighbors);valid=np.ones(nb.shape,bool)
    for j in range(1,nb.shape[1]):valid[:,j]=~(nb[:,:j]==nb[:,j:j+1]).any(1)
    sd=float(score.std())+1e-9;s=score[nb];w=np.exp((s-s.max(1,keepdims=True))/sd)*valid;w/=w.sum(1,keepdims=True)
    points=(w[:,:,None]*geom.points[nb]).sum(1);crop=np.asarray(crop,np.float64);scale=np.maximum(crop[2:]-crop[:2],1)/eye_size;frame=crop[:2]+points*scale
    distances=np.linalg.norm(frame[:,None,:]-centers[None,:,:],axis=2);indices=distances.argmin(0)
    if len(set(indices.tolist()))!=len(indices):raise ValueError('Original anchors cannot be resolved uniquely')
    projected=(score[indices]-score.mean())/sd
    return dict(indices=indices,centers=frame[indices],scores=projected,features=np.asarray([(w[i,:,None]*features[nb[i]]).sum(0) for i in indices]),center_max_px=float(np.abs(frame[indices]-centers).max()),score_max_abs=float(np.abs(projected-scores).max()))
