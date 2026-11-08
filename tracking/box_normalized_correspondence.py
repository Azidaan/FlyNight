"""Observed-box relative coordinates; no claim of physical occlusion alignment."""
from __future__ import annotations
import numpy as np


def box_transform(reference,current):
    a=np.asarray(reference,float);b=np.asarray(current,float)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():return dict(valid=False,reason='invalid_box')
    old=a[2:]-a[:2];new=b[2:]-b[:2]
    if min(*old,*new)<=0:return dict(valid=False,reason='nonpositive_extent')
    return dict(valid=True,scale=(new/old).tolist(),offset=[0.,0.],full_size=new.tolist(),alignment='observed_box_relative_coordinates',physical_part_correspondence_not_established=True)
