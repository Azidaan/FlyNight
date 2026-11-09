"""Fixed template support for candidate-relative search on the same eye maps.

Only the candidate-box clipping gate changes. Reference membership, affine
alignment, physical offsets and all-shift eye-hull validity remain unchanged.
"""
from __future__ import annotations
import numpy as np


def intersection_box(candidate_box, offsets):
    """Exact intersection of B-s for every s, possibly an empty rectangle."""
    box=np.asarray(candidate_box,dtype=np.float64)
    off=np.asarray(offsets,dtype=np.float64)
    if (box.shape!=(4,) or off.ndim!=2 or off.shape[1]!=2 or not len(off)
            or not np.isfinite(np.r_[box,off.ravel()]).all()
            or (box[2:]<=box[:2]).any()):
        raise ValueError('Finite positive candidate box and nonempty2D offsets required')
    return np.r_[box[:2]-off.min(0),box[2:]-off.max(0)]


def relative_template_plan(reader,reference_box,reference_crop,current_box,
                           current_crop,offsets,*,clip_to_candidate=False):
    """Use the full reference box wherever all shifts lie within the eye.

    clip_to_candidate=True reproduces the old single-candidate support for an
    explicit baseline witness. No ID, annotation, score or current image is
    needed to choose cells. The number of cells is fixed across all shifts.
    """
    old=np.asarray(reference_box,dtype=np.float64)
    box=np.asarray(current_box,dtype=np.float64)
    rc=np.asarray(reference_crop,dtype=np.float64)
    ct=np.asarray(current_crop,dtype=np.float64)
    off=np.asarray(offsets,dtype=np.float64)
    if (any(a.shape!=(4,) for a in [old,box,rc,ct])
            or off.ndim!=2 or off.shape[1]!=2 or not len(off)
            or not np.isfinite(np.r_[old,box,rc,ct,off.ravel()]).all()
            or any((a[2:]<=a[:2]).any() for a in [old,box,rc,ct])):
        raise ValueError('Finite positive boxes/crops and nonempty2D offsets required')
    # Preserve the archived arithmetic order for the old-coordinate witness.
    x=reader.points*(rc[2:]-rc[:2])/reader.eye_size+rc[:2]
    trial=box[:2]+(x-old[:2])*(box[2:]-box[:2])/(old[2:]-old[:2])
    trial=trial[:,None]+off
    query=(trial-ct[:2])/((ct[2:]-ct[:2])/reader.eye_size)
    reference_mask=((x>=old[:2])&(x<=old[2:])).all(1)
    eye_mask=(reader.mesh.find_simplex(query)>=0).all(1)
    candidate_mask=((trial>=box[:2])&(trial<=box[2:])).all((1,2))
    keep=reference_mask&eye_mask
    if clip_to_candidate:keep&=candidate_mask
    cells=np.flatnonzero(keep)
    eroded=intersection_box(box,off)
    fraction=float(np.maximum(eroded[2:]-eroded[:2],0).prod()/(box[2:]-box[:2]).prod())
    return dict(cells=cells,query=query[cells],offsets=off,
        reference_cells=int(reference_mask.sum()),
        eye_supported_cells=int((reference_mask&eye_mask).sum()),
        candidate_supported_cells=int((reference_mask&eye_mask&candidate_mask).sum()),
        candidate_intersection_box=eroded,candidate_area_fraction=fraction)
