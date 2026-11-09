"""Experimental spatial reader: visible-edge alignment and channel balancing.

No ID assignment or detector/core changes. Inputs are past/current feature maps
and causal box metadata. Annotation, target IDs and thresholds are not inputs.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay
from tracking.clipping_aware_correspondence import visible_edge_translation
from tracking.channel_balanced_correspondence import channel_balanced_similarity
from tracking.hex_background_filter import unit_patch


class ObservedSpatialReader:
    def __init__(self,points,eye_size,frame_size):
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or self.points.shape[1]!=2 or not np.isfinite(self.points).all() or eye_size<=0:raise ValueError('Finite2D eye geometry required')
        self.eye_size=float(eye_size);self.frame_size=tuple(frame_size);self.mesh=Delaunay(self.points)

    def plan(self,reference_box,reference_crop,current_box,current_crop,prediction,offsets,*,reference_observation=None,use_visible_edge=True):
        """reference_box is the archived state-box support; observation is raw.

        Retaining the old support rule isolates the translation correction.
        Missing raw provenance preserves the prediction instead of pretending
        that the filtered box is an actual detector observation.
        """
        old=np.asarray(reference_box,float);box=np.asarray(current_box,float);c0=np.asarray(reference_crop,float);ct=np.asarray(current_crop,float)
        off=np.asarray(offsets,float);d=np.asarray(prediction,float)
        if any(a.shape!=(4,) for a in [old,box,c0,ct]) or d.shape!=(2,) or off.ndim!=2 or off.shape[1]!=2 or min(*(c0[2:]-c0[:2]),*(ct[2:]-ct[:2]))<=0 or not np.isfinite(np.r_[old,box,c0,ct,d,off.ravel()]).all():raise ValueError('Finite boxes, crops, translation and offsets required')
        anchor=visible_edge_translation(reference_observation,box,d,self.frame_size) if use_visible_edge and reference_observation is not None else dict(translation=d.copy(),changes=[],changed=False)
        d=anchor['translation'];s0=(c0[2:]-c0[:2])/self.eye_size;st=(ct[2:]-ct[:2])/self.eye_size
        x=self.points*s0+c0[:2];trial=x[:,None]+d+off;query=(trial-ct[:2])/st
        keep=((x>=old[:2])&(x<=old[2:])).all(1)&((trial>=box[:2])&(trial<=box[2:])).all((1,2))
        keep&=(self.mesh.find_simplex(query)>=0).all(1);indices=np.flatnonzero(keep)
        return dict(cells=indices,query=query[indices],prediction=d,offsets=off,anchor=anchor)

    def compare(self,reference,current,plan,*,normalization='channels'):
        """Return evidence or abstain at the unchanged three-cell support gate."""
        if normalization not in ['channels','global']:raise ValueError('Unknown normalization')
        a=np.asarray(reference,float);b=np.asarray(current,float)
        if a.ndim!=2 or b.shape!=a.shape or len(a)!=len(self.points) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Equal finite eye maps required')
        idx=np.asarray(plan['cells'],int);query=np.asarray(plan['query'],float)
        if len(idx)<3:return dict(score=None,scores=None,winner=None,reason='fewer_than_three_cells')
        si=self.mesh.find_simplex(query)
        if (si<0).any():raise ValueError('Query outside the eye hull')
        tr=self.mesh.transform[si];w=np.einsum('...ij,...j->...i',tr[...,:2,:],query-tr[...,2,:]);w=np.concatenate([w,1-w.sum(-1,keepdims=True)],-1)
        patches=np.einsum('...v,...vc->...c',w,b[self.mesh.simplices[si]]).transpose(1,0,2)
        if normalization=='channels':scores=[channel_balanced_similarity(a[idx],v) for v in patches]
        else:
            z0=unit_patch(a[idx]);scores=[float(z0@z) if z0 is not None and (z:=unit_patch(v)) is not None else None for v in patches]
        valid=[i for i,s in enumerate(scores) if s is not None];winner=max(valid,key=lambda i:scores[i]) if valid else None
        return dict(score=scores[winner] if winner is not None else None,scores=scores,winner=winner,reason=None if winner is not None else 'constant_patch')
