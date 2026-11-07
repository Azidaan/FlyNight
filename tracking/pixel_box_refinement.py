"""Ограниченная поправка реальной рамки по текущим/прошлым пикселям, без GT во входе."""
import numpy as np

def coordinates(label):
    b=label['box2d'];return np.array([b[k] for k in ['x1','y1','x2','y2']],np.float32)

def region(image,box,size=96,context=1.5):
    import cv2
    x1,y1,x2,y2=map(float,box);w=max(x2-x1,1.);h=max(y2-y1,1.)
    left=(x1+x2-context*w)/2;top=(y1+y2-context*h)/2
    mat=np.array([[size/(context*w),0,-left*size/(context*w)],[0,size/(context*h),-top*size/(context*h)]],np.float32)
    return cv2.warpAffine(image,mat,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))

def apply_offsets(box,offset,alpha=1.,width=1280,height=720):
    b=np.asarray(box,np.float32);d=np.asarray(offset,np.float32)
    if float(alpha)==0 or not np.any(d):return b.copy()
    assert np.isfinite(d).all() and np.max(np.abs(d))<=.250001
    scale=np.array([max(b[2]-b[0],1),max(b[3]-b[1],1)]*2,np.float32)
    out=b+float(alpha)*d*scale;out[[0,2]]=np.clip(out[[0,2]],0,width);out[[1,3]]=np.clip(out[[1,3]],0,height)
    return out if out[2]>out[0]+1 and out[3]>out[1]+1 else b.copy()

def make_model():
    import torch
    from torch import nn
    class Refiner(nn.Module):
        def __init__(self):
            super().__init__();layers=[];a=6
            for b in [24,48,64]:layers += [nn.Conv2d(a,b,3,2,1),nn.GroupNorm(8,b),nn.ReLU()];a=b
            self.vision=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((6,6)),nn.Flatten())
            self.head=nn.Sequential(nn.Linear(64*6*6+12,128),nn.ReLU(),nn.Linear(128,4))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,images,meta,temporal):
            # N,2,3,H,W, first past then current. Current-only has identical parameter budget.
            past,current=images[:,0],images[:,1]
            x=torch.cat([past if temporal else current,current],dim=1)
            return .25*torch.tanh(self.head(torch.cat([self.vision(x),meta],dim=1)))
    return Refiner()

def metadata(label,has_past,dt):
    classes=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle']
    b=coordinates(label);cls=[float(label['category']==c) for c in classes]
    return np.array([float(label.get('score',1)),min((b[2]-b[0])/1280,2),min((b[3]-b[1])/720,2),min(dt/30,1),*cls],np.float32)
