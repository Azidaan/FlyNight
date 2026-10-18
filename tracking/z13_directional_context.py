"""Fixed directional contrasts for a hexagonal per-target linear readout."""
import numpy as np


def directional_neighbors(points, neighbors):
    points=np.asarray(points,dtype=float);neighbors=np.asarray(neighbors,dtype=int)
    edges=[points[j]-points[i] for i,row in enumerate(neighbors) for j in set(row) if j!=i]
    angles=np.arctan2(np.asarray(edges)[:,1],np.asarray(edges)[:,0])
    phase=np.angle(np.mean(np.exp(6j*angles)))/6
    slots=np.repeat(np.arange(len(points))[:,None],6,axis=1)
    for i,row in enumerate(neighbors):
        used=set()
        for j in set(row):
            if j==i:continue
            delta=points[j]-points[i]
            k=int(np.rint((np.arctan2(delta[1],delta[0])-phase)/(np.pi/3)))%6
            if k in used:raise ValueError('Two neighbors assigned to the same direction')
            slots[i,k]=j;used.add(k)
    return slots


def helmert_contrasts():
    h=np.zeros((6,5),dtype=float)
    for k in range(1,6):
        h[:k,k-1]=1/np.sqrt(k*(k+1))
        h[k,k-1]=-k/np.sqrt(k*(k+1))
    return h


def expand_features(standardized, slots):
    """Keep all original68 inputs, append170 zero-sum directional contrasts.

    Uses only the first34 standardized center channels for neighbors.
    The extra block is divided by sqrt5, fixed before any experiment.
    """
    x=np.asarray(standardized,dtype=float);slots=np.asarray(slots,dtype=int)
    if x.shape[-1]!=68 or x.shape[-2]!=len(slots) or slots.shape[1]!=6:
        raise ValueError('Expected68 channels per eye column and6 directional slots')
    local=x[...,slots,:34]
    extra=np.einsum('...nkd,kc->...ncd',local,helmert_contrasts(),optimize=True)
    extra=extra.reshape(*x.shape[:-1],170)/np.sqrt(5.)
    return np.concatenate((x,extra),axis=-1)


def shuffled_neighbors(slots,seed=0):
    rng=np.random.default_rng(seed)
    return np.stack([row[rng.permutation(6)] for row in np.asarray(slots)])
