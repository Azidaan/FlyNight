"""Experimental full-template and two-part correlations at one unchanged shift.

The template, partition and shift never depend on a label or candidate score.
This is a readout candidate, not a deployed identity rule or a FlyVis-core change.
"""
from __future__ import annotations
import numpy as np
from tracking.channel_balanced_correspondence import _center_pair
from tracking.hex_linear_interpolation import sample_linear

def reference_parts(points,cells,reference_box):
    """Two balanced ordered parts along the previous box's longer image axis.

    Returns positions within `cells`, not absolute lattice IDs. Secondary-axis
    and absolute-cell-ID ordering make coordinate ties deterministic. Fewer than
    six cells is undefined; the caller must use its recorded fallback.
    """
    points=np.asarray(points,float);idx=np.asarray(cells,np.int64);box=np.asarray(reference_box,float)
    if (points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all()
            or idx.ndim!=1 or len(np.unique(idx))!=len(idx) or (idx<0).any() or (idx>=len(points)).any()
            or box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=box[:2]).any()):
        raise ValueError('Finite points, unique supported cells and a positive box required')
    if len(idx)<6:return None
    axis=int(np.argmax(box[2:]-box[:2]));order=np.lexsort((idx,points[idx,1-axis],points[idx,axis]))
    middle=len(idx)//2
    return dict(axis=axis,groups=[order[:middle],order[middle:]])

def local_channel_vector(reference,current):
    """Same reference-active channel convention as the existing global reader."""
    x,y,an,bn,active_a,active_b=_center_pair(reference,current)
    if not active_a.any() or not active_b.any():return None
    rho=np.divide((x*y).sum(0),an*bn,out=np.zeros_like(an),where=an*bn>0)
    return rho*reference.shape[1]/int(active_a.sum())

def parts_at_match(reference,current,groups,global_channels):
    a=np.asarray(reference,float);b=np.asarray(current,float);full=np.asarray(global_channels,float)
    if a.ndim!=2 or b.shape!=a.shape or full.shape!=(a.shape[1],) or not np.isfinite(full).all():
        raise ValueError('Aligned patches and the saved full-channel vector required')
    if len(groups)!=2 or sorted(np.concatenate(groups).tolist())!=list(range(len(a))) or min(map(len,groups))<3:
        raise ValueError('Two disjoint supported parts covering the full template required')
    local=[local_channel_vector(a[g],b[g]) for g in groups]
    if any(value is None for value in local):return dict(valid=False,reason='constant_local_patch',channels=None)
    return dict(valid=True,reason=None,channels=np.concatenate([full,*local])/np.sqrt(3),local_channels=local)

def partitioned_channel_features(mesh,reference,current,cells,query,reference_box,baseline):
    """Reuse the saved global vector and winning shift; interpolate only its parts."""
    if not baseline['valid']:return dict(valid=False,reason='undefined_global_match',channels=None)
    idx=np.asarray(cells,np.int64);part=reference_parts(mesh.points,idx,reference_box)
    if part is None:return dict(valid=False,reason='fewer_than_six_cells',channels=None)
    a=np.asarray(reference,float);b=np.asarray(current,float);q=np.asarray(query,float);winner=int(baseline['winner'])
    if (a.ndim!=2 or b.shape!=a.shape or len(a)!=len(mesh.points) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or q.ndim!=3 or q.shape[0]!=len(idx) or q.shape[2]!=2 or not 0<=winner<q.shape[1]
            or not np.isfinite(q).all() or (mesh.find_simplex(q)<0).any()):
        raise ValueError('Finite maps and supported old correspondence required')
    patch=sample_linear(mesh,b,q[:,winner])
    value=parts_at_match(a[idx],patch,part['groups'],baseline['channels'])
    return dict(**value,axis=part['axis'],groups=[idx[g] for g in part['groups']],winner=winner)
