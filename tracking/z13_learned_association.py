"""Small learned partial assignment with explicit unmatched candidates.

Inspired by KeepTrack/SuperGlue; this module does not reproduce their full
networks, images, training distribution, redetection or memory-update policy.
"""
import numpy as np


def pair_inputs(previous, current, width, origin):
    """Only observed candidates; shared coordinate/score units for both frames."""
    if width<=0:raise ValueError('Width must be positive')
    scale=max(float(np.max(np.abs(previous['scores']))),float(np.max(np.abs(current['scores']))),1e-6)
    result=[]
    for f in (previous,current):
        x=np.asarray(f['features'],np.float32)
        g=np.concatenate([(np.asarray(f['centers'])-origin)/width,np.asarray(f['scores'])[:,None]/scale],axis=1).astype(np.float32)
        if not 1<=len(x)<=4 or x.shape!=(len(g),68):raise ValueError('Invalid candidates')
        if not np.isfinite(x).all() or not np.isfinite(g).all():raise ValueError('Nonfinite candidates')
        result.extend([x,g])
    return tuple(result)


def make_association(torch,activity=True):
    nn=torch.nn
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();self.q=nn.Linear(16,16);self.k=nn.Linear(16,16);self.v=nn.Linear(16,16)
            self.out=nn.Linear(16,16);self.norm=nn.LayerNorm(16)
        def forward(self,z,context,mask):
            a=(self.q(z)@self.k(context).transpose(-1,-2)/4).masked_fill(~mask[:,None],-1e4).softmax(-1)
            return self.norm(z+self.out(a@self.v(context)))
    class Association(nn.Module):
        def __init__(self):
            super().__init__();self.activity=activity
            self.embed=nn.Sequential(nn.LayerNorm(68),nn.Linear(68,16),nn.Tanh())
            self.geom=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,16))
            self.layers=nn.ModuleList([Attention() for _ in range(4)])
            self.project=nn.Linear(16,16);self.bin=nn.Parameter(torch.tensor(1.))
        def forward(self,x,g,m,y,h,n):
            if not self.activity:x=torch.zeros_like(x);y=torch.zeros_like(y)
            a=self.embed(x)+self.geom(g);b=self.embed(y)+self.geom(h)
            for i,layer in enumerate(self.layers):
                if i%2==0:a,b=layer(a,a,m),layer(b,b,n)
                else:a,b=layer(a,b,n),layer(b,a,m)
            score=self.project(a)@self.project(b).transpose(-1,-2)/4
            score=score.masked_fill(~(m[:,:,None]&n[:,None,:]),-1e4)
            batch,ns,nt=score.shape
            top=torch.cat([score,self.bin.expand(batch,ns,1)],-1)
            bottom=self.bin.expand(batch,1,nt+1)
            z=torch.cat([top,bottom],-2)
            ms=m.sum(-1).to(score.dtype);nsum=n.sum(-1).to(score.dtype);total=ms+nsum
            logmu=torch.cat([torch.where(m,torch.zeros_like(m,dtype=score.dtype),-1e4),nsum.log()[:,None]],-1)-total.log()[:,None]
            lognu=torch.cat([torch.where(n,torch.zeros_like(n,dtype=score.dtype),-1e4),ms.log()[:,None]],-1)-total.log()[:,None]
            u=torch.zeros_like(logmu);v=torch.zeros_like(lognu)
            for _ in range(10):
                u=logmu-torch.logsumexp(z+v[:,None,:],dim=-1)
                v=lognu-torch.logsumexp(z+u[:,:,None],dim=-2)
            return z+u[:,:,None]+v[:,None,:]+total.log()[:,None,None]
    return Association()


def target_index(label,frame):
    """Training annotation only: canonical closest peak within existing tolerance."""
    if label['kind']==-1:return None
    if label['kind']==0:return -1
    errors=np.linalg.norm(np.asarray(frame['centers'])-label['gt'],axis=1)
    j=int(errors.argmin())
    return j if errors[j]<=.25*label['scale'] else -1


def pack_pairs(pairs):
    """Pad observed sets; assignment labels are stored separately from inputs."""
    size=max(max(len(p[0]),len(p[2])) for p in pairs)
    arrays=[np.zeros((len(pairs),size,68),np.float32),np.zeros((len(pairs),size,3),np.float32),np.zeros((len(pairs),size),bool),
            np.zeros((len(pairs),size,68),np.float32),np.zeros((len(pairs),size,3),np.float32),np.zeros((len(pairs),size),bool)]
    for i,(x,g,y,h) in enumerate(pairs):
        arrays[0][i,:len(x)]=x;arrays[1][i,:len(x)]=g;arrays[2][i,:len(x)]=True
        arrays[3][i,:len(y)]=y;arrays[4][i,:len(y)]=h;arrays[5][i,:len(y)]=True
    return arrays


class LearnedIdentityReadout:
    """Same causal object store/initialization as fixed identity control."""
    def __init__(self,seed_center,seed_features,predict_log_assignment):
        self.origin=np.asarray(seed_center,float);self.objects={0:(self.origin.copy(),np.asarray(seed_features,float),1.)}
        self.next_id=1;self.started=False;self.predict=predict_log_assignment
    def step(self,centers,features,scores,width):
        from scipy.optimize import linear_sum_assignment
        centers=np.asarray(centers);features=np.asarray(features);scores=np.asarray(scores)
        if not self.started:
            d=np.linalg.norm(centers-self.origin,axis=1)/width;j=int(d.argmin()) if d.min()<=.25 else -1
            new={0:self.objects[0]}
            for k,(c,f,s) in enumerate(zip(centers,features,scores)):
                ident=0 if k==j else self.next_id
                if ident:self.next_id+=1
                new[ident]=(c.copy(),f.copy(),float(s))
            self.objects=new;self.started=True;return j,1. if j>=0 else 0.
        keys=list(self.objects);previous=dict(centers=np.stack([self.objects[k][0] for k in keys]),features=np.stack([self.objects[k][1] for k in keys]),scores=np.asarray([self.objects[k][2] for k in keys]))
        current=dict(centers=centers,features=features,scores=scores)
        logits=self.predict(pair_inputs(previous,current,width,self.origin))
        nr,nc=len(keys),len(centers);cost=-logits[:nr,:nc];dummy=np.repeat(-logits[:nr,-1:],nr,axis=1)
        rows,cols=linear_sum_assignment(np.concatenate([cost,dummy],axis=1))
        matched={int(c):int(r) for r,c in zip(rows,cols) if c<nc and logits[r,c]>logits[r,-1]}
        new={};target=-1;confidence=0.
        for j,(c,f,s) in enumerate(zip(centers,features,scores)):
            if j in matched:
                r=matched[j];ident=keys[r]
                if ident==0:target=j;confidence=float(logits[r,j]-logits[r,-1])
            else:ident=self.next_id;self.next_id+=1
            new[ident]=(c.copy(),f.copy(),float(s))
        if 0 not in new:new[0]=self.objects[0]
        self.objects=new;return target,confidence


def predict_pairs(torch,models,pairs):
    arrays=pack_pairs(pairs);t=[torch.as_tensor(a,device='cpu') for a in arrays]
    with torch.no_grad():
        values=[model(*t).cpu().numpy() for model in models]
    z=np.mean(values,axis=0);size=arrays[0].shape[1]
    return [z[i][np.ix_(list(range(len(p[0])))+[size],list(range(len(p[2])))+[size])] for i,p in enumerate(pairs)]


def select_sequences(torch,models,sequences,progress=None,batch_size=128):
    """Batch independent episodes at each time; each object's history stays causal."""
    states={};answers={};lookup={}
    for e,frames in sequences.items():
        f=frames[0];assert f['t']==3
        states[e]=LearnedIdentityReadout(f['seed_center'],f['seed_features'],None)
        answers[(e,3)]=states[e].step(f['centers'],f['features'],f['scores'],f['width'])
        for f in frames[1:]:lookup.setdefault(f['t'],[]).append((e,f))
    for model in models:model.eval()
    for t,items in sorted(lookup.items()):
        if progress:progress(t)
        for start in range(0,len(items),batch_size):
            chunk=items[start:start+batch_size];pairs=[]
            for e,f in chunk:
                state=states[e];keys=list(state.objects)
                previous=dict(centers=np.stack([state.objects[k][0] for k in keys]),features=np.stack([state.objects[k][1] for k in keys]),scores=np.asarray([state.objects[k][2] for k in keys]))
                pairs.append(pair_inputs(previous,f,f['width'],state.origin))
            logits=predict_pairs(torch,models,pairs)
            for (e,f),z in zip(chunk,logits):
                states[e].predict=lambda pair,z=z:z
                answers[(e,t)]=states[e].step(f['centers'],f['features'],f['scores'],f['width'])
    return answers
