"""Service-equivalent batch of native costs; no scientific action change."""
import copy
import numpy as np
from tracking.z13_proposal_geometry import xywh


class FullBoxCosts:
    def __init__(self):
        self.detections=None
        self.costs=None
        self.checked=0

    def __call__(self,track,detections,original,get_dists,choices):
        if self.detections is not detections:
            clones=[]
            for d in detections:
                other=copy.copy(track);other.mean=track.mean.copy();other.mean[:4]=xywh(d.xyxy);clones.append(other)
            self.costs=np.asarray(get_dists(clones,detections)).copy()
            self.detections=detections
        result=[np.asarray(original).copy(),np.ones_like(original),*[self.costs[j].copy() for j in choices[1:]]]
        if self.checked<128:
            from tracking.z13_box_action_labels import full_box_rows
            reference=full_box_rows(track,detections,original,get_dists,choices)
            assert all(np.array_equal(a,b) for a,b in zip(result,reference))
            self.checked+=len(choices)
        return result
