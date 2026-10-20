"""Learnable candidate choice inside the FlyVis readout, with matched ablation.

Inputs are current candidate activity, observed exemplar activity and available
geometry. Neither future labels nor true query centers enter this API.
"""
import numpy as np


def candidate_inputs(centers,features,scores,seed_center,seed_features,pred,width,pos):
    centers=np.asarray(centers,float);features=np.asarray(features,np.float32)
    n=len(centers)
    if not 1<=n<=3 or width<=0 or features.shape!=(n,68):raise ValueError('Invalid candidate input')
    geom=np.zeros((3,8),np.float32);x=np.zeros((3,68),np.float32);mask=np.arange(3)<n
    dist=np.linalg.norm(centers[:,None]-centers[None],axis=2)/width
    np.fill_diagonal(dist,np.inf)
    geom[:n,:2]=(centers-np.asarray(pred))/width
    geom[:n,2:4]=(centers-np.asarray(seed_center))/width
    geom[:n,4]=scores;geom[:n,5]=np.asarray(scores)-scores[0]
    geom[:n,6]=np.minimum(dist.min(1),1.) if n>1 else 1.
    geom[:n,7]=pos/30.;x[:n]=features
    seed=np.asarray(seed_features,np.float32)
    if seed.shape!=(68,) or not all(np.isfinite(v).all() for v in (geom,x,seed)):raise ValueError('Nonfinite candidate input')
    return x,seed,geom,mask


def make_ranker(torch,activity=True):
    nn=torch.nn
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__();self.activity=activity
            self.embed=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh())
            self.score=nn.Sequential(nn.Linear(88,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,seed,geom,mask):
            if not self.activity:x=torch.zeros_like(x);seed=torch.zeros_like(seed)
            z=self.embed(x);s=self.embed(seed)[:,None].expand_as(z)
            context=(z*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
            h=torch.cat([z,s,(z-s).abs(),z*s,context[:,None].expand_as(z),geom],-1)
            return self.score(h).squeeze(-1).masked_fill(~mask,-1e4)
    return Ranker()
