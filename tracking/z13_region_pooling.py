"""Paired mean/max reading of the identical target-response field and regions."""
import numpy as np


def region_probabilities(score,points,boxes,supported,reduction='max'):
    from tracking.z13_region_readout import region_probabilities as mean_probabilities
    if reduction=='mean':return mean_probabilities(score,points,boxes,supported)
    if reduction!='max':raise ValueError('Only preregistered mean or max pooling')
    score=np.asarray(score,np.float64);points=np.asarray(points,np.float64)
    boxes=np.asarray(boxes,np.float64).reshape(-1,4);valid=np.asarray(supported,bool).copy()
    if score.shape!=(len(points),) or points.shape!=(len(score),2) or valid.shape!=(len(boxes),) or not np.isfinite(score).all():raise ValueError('Aligned finite field and original regions required')
    centered=(score-score.mean())/(score.std()+1e-9);logits=np.full(len(boxes),-np.inf)
    for j,box in enumerate(boxes):
        mask=((points>=box[:2])&(points<=box[2:])).all(1);valid[j]&=bool(mask.any())
        if valid[j]:logits[j]=float(centered[mask].max())
    offset=max(0.,float(logits.max(initial=-np.inf)));weights=np.exp(logits-offset)
    return weights/(np.exp(-offset)+weights.sum()),valid
