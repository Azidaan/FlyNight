"""Absolute spatial compatibility with one track; no competitor is required."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from tracking.spatial_identity_assignment import frame_points,centered_similarity,flow_similarity


def correspondence(transform,history,current_crop,box,geometry,current_valid,reference_valid):
    if not transform.get('valid'):return None
    box=np.asarray(box,float);old=np.asarray(history[-1]['box'],float);crop0=np.asarray(history[-1]['crop'],float)
    p0=frame_points(geometry['points'],crop0,geometry['eye_size']);pt=frame_points(geometry['points'],current_crop,geometry['eye_size'])
    mask=((p0>=old[:2])&(p0<=old[2:])).all(1);ids=np.flatnonzero(((pt>=box[:2])&(pt<=box[2:])).all(1))
    projected=old[:2]+transform['offset']+(pt[ids]-box[:2])/transform['scale'];distance,qq=cKDTree(p0).query(projected)
    spacing=geometry['spacing']*(crop0[2]-crop0[0])/geometry['eye_size']
    keep=((projected>=old[:2])&(projected<=old[2:])).all(1)&mask[qq]&(distance<=.6*spacing)
    pp=ids[keep];qq=qq[keep];keep=np.asarray(current_valid,bool)[pp]&np.asarray(reference_valid,bool)[qq];pp=pp[keep];qq=qq[keep]
    unique,inverse=np.unique(qq,return_inverse=True)
    if len(unique)<3:return None
    return dict(p=pp,q=qq,unique=unique,inverse=inverse,reference_ids=np.flatnonzero(mask&np.asarray(reference_valid,bool)))


def gather(current,pairs):
    current=np.asarray(current,float);x=np.zeros((len(pairs['unique']),current.shape[1]))
    np.add.at(x,pairs['inverse'],current[pairs['p']]);return x/np.bincount(pairs['inverse'])[:,None]


def absolute_score(kind,current,reference,pairs,scale,*,video,tid,last_frame,frame,column):
    if pairs is None:return None
    if kind=='SUPPORT':return 1.
    if kind=='RANDOM':
        value=int.from_bytes(hashlib.sha256(f'{video}|{frame}|{tid}|{column}|42'.encode()).digest()[:8],'little')
        return 2*(value/2**64)-1
    x=gather(current,pairs);y=np.asarray(reference,float);q=pairs['unique']
    if kind=='DIS':return flow_similarity(x,y[q],scale)
    if kind!='PERM':return centered_similarity(x,y[q])
    values=[];ids=pairs['reference_ids']
    for rep in range(32):
        seed=int.from_bytes(hashlib.sha256(f'{video}|{tid}|{last_frame}|{rep}|42'.encode()).digest()[:8],'little')
        permuted=y.copy();permuted[ids]=y[np.random.default_rng(seed).permutation(ids)]
        values.append(centered_similarity(x,permuted[q]))
    return float(np.mean(values)) if all(v is not None for v in values) else None


def passes(value,threshold):
    return threshold!='off' and value is not None and bool(np.isfinite(value)) and value>=threshold
