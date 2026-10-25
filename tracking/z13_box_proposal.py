"""Target-response readout -> actual region proposal -> center or full box state."""
import numpy as np
from tracking.z13_region_readout import region_probabilities
from tracking.z13_proposal_geometry import xywh


def choose_region(field, points, eye_boxes, supported):
    probabilities, valid = region_probabilities(field, points, eye_boxes, supported)
    null = max(0., 1-float(probabilities.sum()))
    # NULL is first and wins exact ties. No additional confidence threshold.
    index = int(np.argmax(np.r_[null, probabilities])) - 1
    if index < 0 or not valid[index]:
        return -1, probabilities, valid
    return index, probabilities, valid


def proposal_costs(track, detections, original, get_dists, selected, alphas, full_box):
    original=np.asarray(original).copy();saved=track.mean.copy();out=[]
    try:
        for alpha in alphas:
            if not 0 <= alpha <= 1:
                raise ValueError('Interpolation weight outside [0,1]')
            if alpha==0 or selected<0:
                out.append(original.copy());continue
            target=xywh(detections[selected].xyxy)
            track.mean[:]=saved
            length=4 if full_box else 2
            track.mean[:length]=(1-alpha)*saved[:length]+alpha*target[:length]
            out.append(np.asarray(get_dists([track],detections)[0]).copy())
    finally:
        track.mean[:]=saved
    assert np.array_equal(saved,track.mean)
    return out
