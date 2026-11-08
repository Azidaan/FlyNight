"""Ordinary-box singleton eligibility; no retinal support or detection-count gate."""
from __future__ import annotations
import numpy as np


def iou(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():return 0.
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(size.prod())
    area=float(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter)
    return inter/area if area>0 else 0.


def proposals(event,*,cap=6,threshold=.2,bidirectional=False):
    """Only one forward-eligible free box; a backward veto never creates a singleton.

    Candidates are already same-class/free after native first association.
    A first-public native confirmation is mandatory. Other eligible free boxes
    still compete even if they do not produce a first public confirmation.
    """
    if cap<0 or not 0<threshold<=1:raise ValueError('Invalid bounded geometry configuration.')
    result=[]
    for r in event['rows']:
        if r['age']>cap:continue
        eligible=[c for c in r['candidates'] if c['forward_iou']>=threshold]
        if len(eligible)!=1:continue
        c=eligible[0]
        if c['new_tid'] is None:continue
        backward=None
        if bidirectional:
            birth=c['birth'];dt=event['frame']-birth['frame']
            if dt!=1:raise ValueError('Expected immediately preceding unconfirmed birth.')
            current=np.asarray(c['box'],float);previous=np.asarray(birth['box'],float)
            velocity=((current[:2]+current[2:])-(previous[:2]+previous[2:]))/(2*dt)
            projected=current-np.tile(velocity,2)*r['age'];backward=iou(projected,r['row_data']['history'][-1]['box'])
            if backward<threshold:continue
        result.append(dict(old_tid=r['old_tid'],new_tid=c['new_tid'],cost=c['cost'],age=r['age'],row_data=r['row_data'],forward_iou=c['forward_iou'],backward_iou=backward,column=c['column']))
    return result
