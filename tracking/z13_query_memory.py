"""Query-conditioned reading of separately fitted observed target models.

This light mixture of ridge experts borrows the query-memory reading principle,
not STMTrack's full architecture. No query labels or true centers enter the API.
"""
import numpy as np
from tracking.z13_observed_memory import observation_slots
from tracking.z13_target_center import ridge_scores,observed_center_target


def expert_maps(observations,masks,search,frames,points,spacing,lam):
    observations=np.asarray(observations,dtype=np.float64);masks=np.asarray(masks,bool)
    if observations.shape[:2]!=masks.shape or len(observations)!=4:raise ValueError('Invalid observed bank')
    slots=observation_slots(frames)
    maps=[ridge_scores(observations[t],observed_center_target(masks[t],points,spacing),search,[lam])[0] for t in slots]
    return np.asarray(maps),slots


def fuse(maps,mode,eta=0.,temperature=1.):
    maps=np.asarray(maps,dtype=np.float64)
    if maps.ndim<2 or not 1<=len(maps)<=3 or not np.isfinite(maps).all():raise ValueError('Invalid expert maps')
    if not 0<=eta<=1 or temperature<=0 or mode not in ('last','fixed','query'):raise ValueError('Invalid fusion')
    if mode=='last' or eta==0 or len(maps)==1:return maps[-1].copy()
    if mode=='fixed':memory=maps.mean(0)
    else:
        z=(maps-maps.mean(-1,keepdims=True))/(maps.std(-1,keepdims=True)+1e-9)
        z=z/temperature;z-=z.max(0,keepdims=True)
        w=np.exp(z);w/=w.sum(0,keepdims=True);memory=(w*maps).sum(0)
    return (1-eta)*maps[-1]+eta*memory
