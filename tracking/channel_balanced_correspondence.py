"""Spatial correspondence with explicit cell/channel attribution.

The experimental reader gives each nonconstant REFERENCE channel equal weight.
Current-channel disappearance yields zero evidence, not a smaller denominator.
It changes no core activity, geometric support, shift grid or tracker cost.
"""
from __future__ import annotations
import numpy as np


def _center_pair(reference, current):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(current,dtype=np.float64)
    if a.ndim!=2 or b.shape!=a.shape or min(a.shape)<1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Equal finite cells-by-channels matrices required')
    # Exact constant detection avoids a floating-point mean residual turning an
    # exactly constant channel into artificial evidence. No learned noise cutoff.
    active_a=np.ptp(a,axis=0)!=0;active_b=np.ptp(b,axis=0)!=0
    x=a-a.mean(axis=0);y=b-b.mean(axis=0)
    x[:,~active_a]=0;y[:,~active_b]=0
    an=np.linalg.norm(x,axis=0);bn=np.linalg.norm(y,axis=0)
    return x,y,an,bn,active_a,active_b


def channel_balanced_similarity(reference,current):
    """Mean spatial Pearson correlation over reference-active channels.

    None means an entirely constant reference/current patch. A channel which
    becomes constant in the candidate contributes zero but stays in the count.
    Exactly zero reference channels do not change the result when appended.
    """
    x,y,an,bn,active_a,active_b=_center_pair(reference,current)
    if not active_a.any() or not active_b.any():return None
    denominator=an*bn
    rho=np.divide((x*y).sum(axis=0),denominator,out=np.zeros_like(an),where=denominator>0)
    return float(rho.sum()/active_a.sum())


def centered_contributions(reference,current):
    """Additive explanation of the old score at ONE already chosen shift.

    S=sum_pc d_pc=sum_c w_c*rho_c. It is an accounting identity, not a causal
    neuron intervention: the common centering and denominator couple all cells.
    """
    x,y,an,bn,active_a,active_b=_center_pair(reference,current)
    norm_a=np.linalg.norm(x);norm_b=np.linalg.norm(y)
    if norm_a==0 or norm_b==0:return None
    d=x*y/(norm_a*norm_b);w=an*bn/(norm_a*norm_b)
    rho=np.divide((x*y).sum(0),an*bn,out=np.zeros_like(an),where=an*bn>0)
    balanced=np.divide(x*y,an*bn,out=np.zeros_like(x),where=(an*bn)[None]>0)/active_a.sum()
    return dict(score=float(d.sum()),cell_channel=d,cell=d.sum(1),channel=d.sum(0),
        channel_cosine=rho,channel_amplitude_weight=w,
        reference_channel_energy=an**2/norm_a**2,current_channel_energy=bn**2/norm_b**2,
        reference_cell_energy=(x*x).sum(1)/norm_a**2,current_cell_energy=(y*y).sum(1)/norm_b**2,
        reference_active=active_a,current_active=active_b,
        balanced_score=float(balanced.sum()),balanced_cell=balanced.sum(1),balanced_channel=balanced.sum(0))


def channel_difference_attribution(own,foreign):
    """Exact symmetric two-factor identity for q_foreign-q_own=w*rho.

    The two terms describe changed spatial correlation and changed relative
    channel weight. This is algebraic bookkeeping, not a neuron-causality claim.
    """
    wo=np.asarray(own['channel_amplitude_weight']);wf=np.asarray(foreign['channel_amplitude_weight'])
    ro=np.asarray(own['channel_cosine']);rf=np.asarray(foreign['channel_cosine'])
    if len({x.shape for x in [wo,wf,ro,rf]})!=1:raise ValueError('Aligned channel types required')
    spatial=(wf+wo)*(rf-ro)/2
    amplitude=(rf+ro)*(wf-wo)/2
    return dict(spatial=spatial,amplitude=amplitude,total=wf*rf-wo*ro)
