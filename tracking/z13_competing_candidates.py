"""Loss-only joint competition of candidate actions and no useful candidate."""
import math
import numpy as np


def marginal_choice_loss(torch,scores,targets,mask,positive_weight):
    if scores.shape!=targets.shape or scores.shape!=mask.shape or scores.ndim!=2 or not bool(mask.any(1).all()):raise ValueError('Nonempty masked candidate sets required')
    s=scores.masked_fill(~mask,-torch.inf);good=(targets>0)&mask;has=good.any(1)
    denominator=torch.logsumexp(torch.cat([s,torch.zeros_like(s[:,:1])],1),1)
    # Avoid all-inf derivatives on null-only rows; the chosen numerator is zero.
    positive=s.masked_fill(~good,-torch.inf)
    positive=torch.where(has[:,None],positive,torch.zeros_like(positive))
    numerator=torch.where(has,torch.logsumexp(positive,1),torch.zeros_like(denominator))
    weight=torch.where(has,torch.full_like(denominator,float(positive_weight)),torch.ones_like(denominator))
    return ((denominator-numerator)*weight).mean()


def choice_rows(labels,arrays,logits):
    from scripts.z13_candidate_ranker_20261021 import make_rows
    rows=make_rows(labels,arrays,logits)
    s=np.asarray(logits,np.float64);m=np.maximum(s.max(1),0.);den=np.exp(-m)+np.exp(s-m[:,None]).sum(1)
    for i,r in enumerate(rows):r['p_present']=float(np.exp(s[i,r['candidate']]-m[i])/den[i])
    return rows


def quota_precision(rows,q=.05):
    groups={}
    for row in rows:groups.setdefault(row['video'],[]).append(row)
    values={};details={}
    for video,items in groups.items():
        k=max(1,math.ceil(q*len(items)));selected=sorted(items,key=lambda r:-r['p_present'])[:k]
        known=[r for r in selected if r['kind']>=0];useful=sum(bool(r.get('useful',False)) for r in known)
        if known:values[video]=useful/len(known)
        details[video]=dict(all_rows=len(items),selected=k,known=len(known),unknown=k-len(known),useful=useful,known_nonuseful=len(known)-useful,strict_center_worse=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected),known_unavailable=sum(r['kind']==0 for r in selected))
    return values,details
