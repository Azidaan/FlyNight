"""Actual full-XYWH actions, retaining unchanged and explicit unmatched choices."""
import numpy as np
from tracking.z13_box_proposal import proposal_costs


def full_box_rows(track,detections,original,get_dists,choices):
    original=np.asarray(original).copy()
    if not choices or choices[0]!=-1 or any(j<0 or j>=len(detections) for j in choices[1:]):
        raise ValueError('One explicit unmatched choice followed by real region indices required')
    rows=[original.copy(),np.ones_like(original)]
    for j in choices[1:]:
        rows.append(proposal_costs(track,detections,original,get_dists,j,[1.],True)[0])
    return rows
