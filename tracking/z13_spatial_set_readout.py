"""Internal region readout of all supported (activity, relative-position) pairs."""
import numpy as np


def region_coordinates(points, masks, spacing):
    points=np.asarray(points,dtype=np.float64);masks=np.asarray(masks,dtype=bool)
    if points.ndim!=2 or points.shape[1]!=2 or masks.ndim!=2 or masks.shape[1]!=len(points) or not masks.any(1).all() or spacing<=0:
        raise ValueError('Nonempty observed retinal support required')
    lo=np.stack([points[m].min(0) for m in masks]);hi=np.stack([points[m].max(0) for m in masks])
    return ((lo+hi)/2).astype(np.float32),np.maximum(hi-lo,spacing).astype(np.float32)


def make_model(torch, mode='spatial', activity=True):
    if mode not in ('spatial','mean'):raise ValueError(mode)
    nn=torch.nn
    class SpatialSetReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.point=nn.Sequential(nn.Linear(70,16),nn.Tanh())
            self.target=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh())
            self.score=nn.Sequential(nn.Linear(94,32),nn.ReLU(),nn.Linear(32,1))
            self.default=nn.Sequential(nn.Linear(46,32),nn.ReLU(),nn.Linear(32,1))

        def forward(self,x,seed,geo,mask,full,retinal_mask,centers,scales,points):
            b,k,n=(retinal_mask & mask[...,None]).nonzero(as_tuple=True)
            B,K=mask.shape;ids=b*K+k
            if mode=='spatial':values=full[b,n]
            else:values=x[b,k]
            if not activity:values=torch.zeros_like(values);seed=torch.zeros_like(seed)
            coord=(points[n]-centers[b,k])/scales[b,k]
            encoded=self.point(torch.cat([values,coord],-1))
            sums=torch.zeros((B*K,16),dtype=encoded.dtype,device=encoded.device).index_add(0,ids,encoded)
            counts=torch.zeros(B*K,dtype=encoded.dtype,device=encoded.device).index_add(0,ids,torch.ones_like(ids,dtype=encoded.dtype))
            z=(sums/counts.clamp(min=1)[:,None]).reshape(B,K,16)
            target=self.target(seed);t=target[:,None].expand_as(z)
            denom=mask.sum(1,keepdim=True).clamp(min=1)
            pooled=(z*mask[...,None]).sum(1)/denom
            pg=(geo*mask[...,None]).sum(1)/denom
            h=torch.cat([z,t,(z-t).abs(),z*t,pooled[:,None].expand_as(z),geo],-1)
            return (self.score(h).squeeze(-1)-self.default(torch.cat([pooled,target,pg],-1))).masked_fill(~mask,-1e4)
    return SpatialSetReadout()
