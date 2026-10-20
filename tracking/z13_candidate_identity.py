"""Small causal candidate identity readout, inspired by KeepTrack.

No learned GNN, no full KeepTrack reproduction. Every call sees observations
through the current frame only; labels and future images are absent from API.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


class IdentityReadout:
    def __init__(self, seed_center, seed_features, alpha, missing_cost):
        if not 0<=alpha<=1 or missing_cost<=0:raise ValueError('Invalid matching parameters')
        self.alpha=float(alpha);self.missing=float(missing_cost)
        self.objects={0:(np.asarray(seed_center,float),np.asarray(seed_features,float))}
        self.next_id=1;self.started=False

    @staticmethod
    def appearance(a,b):
        n=np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None]
        return (1-np.clip((a@b.T)/np.maximum(n,1e-12),-1,1))/2

    def step(self, centers, features, crop_width):
        centers=np.asarray(centers,float);features=np.asarray(features,float)
        if centers.ndim!=2 or centers.shape[1]!=2 or features.ndim!=2 or len(centers)!=len(features):raise ValueError('Invalid candidate shape')
        if crop_width<=0 or not np.isfinite(centers).all() or not np.isfinite(features).all():raise ValueError('Invalid candidate values')
        if not self.started:
            # The observed target box seeds its nearest peak if it is nearby.
            # All other current peaks are distractors, not negative GT labels.
            d=np.linalg.norm(centers-self.objects[0][0],axis=1)/crop_width
            target=int(np.argmin(d)) if len(d) and d.min()<=.25 else -1
            new={0:self.objects[0]}
            for j,(c,f) in enumerate(zip(centers,features)):
                ident=0 if j==target else self.next_id
                if ident:self.next_id+=1
                new[ident]=(c.copy(),f.copy())
            self.objects=new;self.started=True
            return target,1. if target>=0 else 0.
        keys=list(self.objects);prev=np.stack([self.objects[i][0] for i in keys]);pf=np.stack([self.objects[i][1] for i in keys])
        dist=np.minimum(np.linalg.norm(prev[:,None]-centers[None],axis=2)/crop_width,1.)
        cost=self.alpha*self.appearance(pf,features)+(1-self.alpha)*dist
        dummy=np.full((len(keys),len(keys)),self.missing)
        rr,cc=linear_sum_assignment(np.concatenate([cost,dummy],axis=1))
        assignments={int(c):int(r) for r,c in zip(rr,cc) if c<len(centers) and cost[r,c]<self.missing}
        new={};target=-1;confidence=0.
        for j,(c,f) in enumerate(zip(centers,features)):
            if j in assignments:
                r=assignments[j];ident=keys[r]
                if ident==0:target=j;confidence=max(0.,1-cost[r,j]/self.missing)
            else:ident=self.next_id;self.next_id+=1
            new[ident]=(c.copy(),f.copy())
        # A missing target keeps its last observed representation for possible
        # redetection. Distractors are propagated only when actually matched.
        if 0 not in new:new[0]=self.objects[0]
        self.objects=new
        return target,float(confidence)
