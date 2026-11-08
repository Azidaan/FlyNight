"""Use a shared visible box edge when the image border clips the other edge.

Only the affected translation axis changes. All other predicted translation
components, signal maps and matching rules stay outside this helper.
"""
from __future__ import annotations
import numpy as np


def visible_edge_translation(reference_detection,current_detection,predicted_translation,frame_size):
    a=np.asarray(reference_detection,dtype=float);b=np.asarray(current_detection,dtype=float)
    d=np.asarray(predicted_translation,dtype=float).copy();size=np.asarray(frame_size,dtype=float)
    if a.shape!=(4,) or b.shape!=(4,) or d.shape!=(2,) or size.shape!=(2,) or not np.isfinite(np.r_[a,b,d,size]).all() or min(size)<=0 or min(*(a[2:]-a[:2]),*(b[2:]-b[:2]))<=0:
        raise ValueError('Finite positive boxes, translation and frame dimensions required')
    changes=[]
    for axis,extent in enumerate(size):
        lower_a=a[axis]<=0;lower_b=b[axis]<=0;upper_a=a[axis+2]>=extent;upper_b=b[axis+2]>=extent
        if not any([lower_a,lower_b,upper_a,upper_b]):continue
        if (upper_a or upper_b) and not (lower_a or lower_b):
            edge=axis;name='lower_visible_edge'
        elif (lower_a or lower_b) and not (upper_a or upper_b):
            edge=axis+2;name='upper_visible_edge'
        else:
            changes.append(dict(axis=axis,changed=False,reason='No common unclipped edge; preserve existing prediction'))
            continue
        before=float(d[axis]);d[axis]=b[edge]-a[edge]
        changes.append(dict(axis=axis,changed=bool(d[axis]!=before),edge=name,old=before,new=float(d[axis])))
    return dict(translation=d,changes=changes,changed=any(c['changed'] for c in changes))
