"""Read a visible reference template in the admission's existing coordinates.

The transform is supplied by geometric_match, not estimated from feature scores.
Reference support is fixed before all shifts. This experimental reader changes
neither FlyVis dynamics nor association costs or the geometric admission rule.
"""
from __future__ import annotations
import numpy as np


def geometric_template_plan(reader, reference_box, reference_crop, current_box,
                            current_crop, offsets, scale, offset):
    """F(x)=current_lo+scale*(x-reference_lo-offset), with visible fixed support."""
    old,rc,box,ct=[np.asarray(v,dtype=np.float64) for v in
                   (reference_box,reference_crop,current_box,current_crop)]
    shifts=np.asarray(offsets,dtype=np.float64)
    s=np.asarray(scale,dtype=np.float64);o=np.asarray(offset,dtype=np.float64)
    if (any(v.shape!=(4,) for v in (old,rc,box,ct)) or s.shape!=(2,) or o.shape!=(2,)
            or shifts.ndim!=2 or shifts.shape[1]!=2 or len(shifts)==0
            or not np.isfinite(np.r_[old,rc,box,ct,s,o,shifts.ravel()]).all()
            or any((v[2:]<=v[:2]).any() for v in (old,rc,box,ct))
            or (s<=0).any() or (o<0).any()):
        raise ValueError('Finite positive boxes/crops/scale, nonnegative reference offset and2Dshifts required')
    x=reader.points*(rc[2:]-rc[:2])/reader.eye_size+rc[:2]
    nominal=box[:2]+s*(x-old[:2]-o)
    query=(nominal[:,None,:]+shifts[None,:,:]-ct[:2])/((ct[2:]-ct[:2])/reader.eye_size)
    reference=((x>=old[:2])&(x<=old[2:])).all(1)
    visible=((nominal>=box[:2])&(nominal<=box[2:])).all(1)
    eye=(reader.mesh.find_simplex(query)>=0).all(1)
    cells=np.flatnonzero(reference&visible&eye)
    return dict(cells=cells,query=query[cells],offsets=shifts.copy(),
                scale=s.copy(),offset=o.copy(),reference_cells=int(reference.sum()),
                visible_reference_cells=int((reference&visible).sum()),
                eye_supported_cells=int((reference&visible&eye).sum()))
