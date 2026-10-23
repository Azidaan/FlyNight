"""Observed-only target-model descriptor and matched dual-embedding ranker."""
import numpy as np


def fit_descriptor(x, target, lam):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or lam<=0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid observed-only regression inputs')
    mu=x.mean(0);xc=x-mu;ym=y.mean()
    w=np.linalg.solve(xc.T@xc/len(x)+lam*np.eye(x.shape[1]),xc.T@(y-ym)/len(x))
    return w,mu,float(ym)


def make_ranker(torch,activity=True):
    nn=torch.nn
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__();self.activity=activity
            self.embed=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh())
            self.target=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh())
            self.score=nn.Sequential(nn.Linear(88,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,seed,geom,mask):
            if not self.activity:x=torch.zeros_like(x);seed=torch.zeros_like(seed)
            z=self.embed(x);s=self.target(seed)[:,None].expand_as(z)
            context=(z*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
            h=torch.cat([z,s,(z-s).abs(),z*s,context[:,None].expand_as(z),geom],-1)
            return self.score(h).squeeze(-1).masked_fill(~mask,-1e4)
    return Ranker()
