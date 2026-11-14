"""Non-mutating, two-sided ownership assignment for saved mask observations."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass(frozen=True)
class Assignment:
    pairs: tuple
    deferred: tuple
    scores: tuple

def edges(scores, owner_ids, observation_ids, *, protected=None):
    """Cost is 1-IoU. Dummy choices always permit deferral.

    The global objective maximizes the sum of overlap above 0.5. Return only
    edges which are strict best choices in both directions. Exact ties and
    unavailable inputs defer; do not substitute the order of the lists.
    """
    a=np.array(scores,dtype=float,copy=True)
    if a.shape!=(len(owner_ids),len(observation_ids)) or not np.all(np.isfinite(a)):raise ValueError('invalid evidence matrix')
    if len(set(owner_ids))!=len(owner_ids) or len(set(observation_ids))!=len(observation_ids):raise ValueError('identifiers must be unique')
    if np.any((a<0)|(a>1)):raise ValueError('overlap must lie between zero and one')
    occupied=protected or {};allowed=np.ones(a.shape,bool)
    for j,obs in enumerate(observation_ids):
        if obs in occupied:
            for i,own in enumerate(owner_ids):allowed[i,j]=occupied[obs]==own
    # Occupied observations still participate in the mutual-competition check.
    strict=np.zeros_like(allowed)
    for i in range(len(owner_ids)):
        for j in range(len(observation_ids)):
            strict[i,j]=a[i,j]>.5 and np.all(a[i,j]>np.delete(a[i],j)) and np.all(a[i,j]>np.delete(a[:,j],i))
    feasible=allowed&strict;utility=np.where(feasible,a-.5,0)
    rr,cc=linear_sum_assignment(-np.c_[utility,np.zeros((len(owner_ids),len(owner_ids)))])
    pairs=tuple((owner_ids[i],observation_ids[j]) for i,j in zip(rr,cc) if j<a.shape[1] and feasible[i,j]);used={q[0] for q in pairs}
    return Assignment(pairs,tuple(q for q in owner_ids if q not in used),tuple(tuple(float(x) for x in line) for line in a))

def overlap_matrix(predicted,observations):
    """Compare visible pixels without access to GT, IDs, color, or depth."""
    output=np.zeros((len(predicted),len(observations)))
    for i,x in enumerate(predicted):
        x=np.asarray(x,bool)
        for j,y in enumerate(observations):
            y=np.asarray(y,bool)
            if x.shape!=y.shape:raise ValueError('spatial grids differ')
            intersection=np.count_nonzero(x&y);union=np.count_nonzero(x|y)
            output[i,j]=intersection/union if union else 0
    return output
