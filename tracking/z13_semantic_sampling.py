"""Training-only semantic distractor audit and matched-budget sampling."""
import numpy as np
from tracking.z13_matched_data_budget import matched_batches


def semantic_negatives(centers,label,target_id,target_category,objects):
    """A peak uniquely inside another same-class box, outside every target box.

Only known outcomes are eligible. Existing useful-center positives are never
reclassified. Objects are annotation tuples (identity, category, xyxy).
"""
    centers=np.asarray(centers,float);result=np.zeros(len(centers),bool)
    if label['kind']<0 or not target_id or target_category is None:return result
    useful=np.zeros(len(centers),bool)
    if label['kind']==1:
        error=np.linalg.norm(centers-label['gt'],axis=1);pred_error=np.linalg.norm(np.asarray(label['pred'])-label['gt'])
        useful=(error<=.25*label['scale'])&(error<pred_error)
    for j,c in enumerate(centers):
        if useful[j]:continue
        covering=[(i,k) for i,k,b in objects if b[0]<=c[0]<=b[2] and b[1]<=c[1]<=b[3]]
        if len(covering)==1:
            identity,category=covering[0]
            result[j]=identity!=target_id and category==target_category
    return result


def semantic_batches(n,nref,seed,hard_indices):
    """Half uniform, half hard rows; same1150 updates and145950 presentations."""
    hard=np.asarray(hard_indices,np.int64)
    if len(hard)==0 or hard.min()<0 or hard.max()>=n:raise ValueError('Nonempty valid hard subset required')
    rng=np.random.default_rng(seed+10000)
    for original in matched_batches(n,nref,50,128,seed):
        batch=original.copy();count=len(batch)//2
        # Preserve the uniform generator and its full consumption exactly.
        positions=rng.choice(len(batch),size=count,replace=False)
        batch[positions]=rng.choice(hard,size=count,replace=True)
        yield batch
