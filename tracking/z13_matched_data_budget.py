"""Change training data volume while preserving optimizer/sample budgets."""
import numpy as np


def matched_batches(n_train,n_reference,epochs=50,batch_size=128,seed=0):
    if min(n_train,n_reference,epochs,batch_size)<=0:raise ValueError('Positive sizes required')
    rng=np.random.default_rng(seed);remaining=np.empty(0,np.int64)
    for _ in range(epochs):
        for left in range(0,n_reference,batch_size):
            want=min(batch_size,n_reference-left);parts=[]
            while want:
                if not len(remaining):remaining=rng.permutation(n_train)
                take=min(want,len(remaining));parts.append(remaining[:take]);remaining=remaining[take:];want-=take
            yield np.concatenate(parts)
