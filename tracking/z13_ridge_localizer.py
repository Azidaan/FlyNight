"""Closed-loop use of the already validated observed-center ridge readout.

No annotation inputs. Only lost-track centers may change; observed masks,
four-frame histories, normalizers and ridge lambdas use the frozen protocol.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np
from tracking.z13_readout import Localizer,HexGeometry,FlyBackbone,N_EXEMPLAR,to_frame
from tracking.z13_target_center import observed_center_target


def fit_ridge(features,target,lam):
    x=np.asarray(features,np.float64);y=np.asarray(target,np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or lam<=0 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid observed regression input')
    mu=x.mean(0);xc=x-mu;bias=float(y.mean());yc=y-bias
    weight=np.linalg.solve(xc.T@xc/len(x)+lam*np.eye(x.shape[1]),xc.T@yc/len(x))
    return mu,weight,bias


def ridge_response(model,features):
    mu,weight,bias=model
    return (np.asarray(features,np.float64)-mu)@weight+bias


def augment(features,neighbors):
    f=np.asarray(features,np.float64)
    return np.concatenate([f,f[:,:,neighbors[:,1:]].mean(-1)],1)


R1_TAUS=np.geomspace(1.5,32.,17)
def receptor_step(ema,current):
    h=np.asarray(current,np.float64)[:,None,:]
    next_ema=np.asarray(ema,np.float64)+(h-ema)*(1.0/R1_TAUS)[None,:,None]
    return next_ema,np.concatenate([next_ema,h-next_ema],1)


def map_center(score,geom):
    s=np.asarray(score,np.float64);k=int(np.argmax(s));nb=np.unique(geom.neighbors[k]);sd=float(s.std())+1e-9
    w=np.exp((s[nb]-s[nb].max())/sd)
    return (geom.points[nb]*w[:,None]).sum(0)/w.sum(),float((s.max()-s.mean())/sd)


class CenterRidgeLocalizer(Localizer):
    def __init__(self,config_path,bank,frame_size=(1280,720)):
        import torch
        from tracking.flyvis_motion import integration_plan
        config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        if config['format']!='z13-center-ridge-hota-v1':raise ValueError('Explicit fixed ridge policy required')
        self.torch,self.bank,self.frame_size,self.deque=torch,bank,frame_size,deque
        self.arm=config['arm'];self.tau=config['tau'];self.lam=float(config['lambda_value']);self.max_batch=4
        self.geom=HexGeometry(bank.points_local,bank.eye_size);self.dev=bank.estimator.flyvis.device
        self.mean=np.asarray(config['normalization']['mean'],np.float64);self.sd=np.asarray(config['normalization']['sd'],np.float64)
        assert self.mean.shape==self.sd.shape==(68,) and (self.sd>0).all()
        self.bb=None
        if self.arm in ('C','D'):
            if self.arm=='D':
                root=Path(__file__).resolve().parents[1];path=(root/config['backbone_file']).resolve();path.relative_to(root)
                ck=torch.load(path,map_location='cpu',weights_only=False)
                bank.estimator.network.load_state_dict({k:v.to(self.dev) for k,v in ck['backbone'].items()})
            self.bb=FlyBackbone(bank.estimator,trainable=False);self.bb.steps,self.bb.dt=integration_plan(30.,.02)
        elif self.arm!='R1':raise ValueError('Supported fixed arms: C/D/R1')
        self.hist,self.eyes={},{}
        self.stats=dict(steps=0,interventions=0,refusals=0,births=0,nonfinite_refusals=0)

    def features(self,raw):
        f=augment(raw,self.geom.neighbors)
        return (f.transpose(0,2,1)-self.mean)/self.sd

    def start_sequence(self,hexes,masks):
        """Same four saved observations as the offline protocol; returns per-eye state."""
        torch=self.torch;hx=hexes.half().float();batch=len(hx)
        if self.arm=='R1':
            h=hx.cpu().numpy().astype(np.float64);state=np.repeat(h[:,0,None,:],17,axis=1)
            for j in range(4):state,f=receptor_step(state,h[:,j])
            packed=[x.copy() for x in state]
        else:
            state=self.bb.init(hx[:,:1]);last=hx[:,0]
            for j in range(1,4):state,f=self.bb.step(state,last,hx[:,j]);last=hx[:,j]
            f=f.double().cpu().numpy();p=self._pack(state);packed=[x.clone() for x in p]
        normalized=self.features(f);models=[]
        for i in range(batch):
            target=observed_center_target(masks[i],self.geom.points,self.geom.spacing);models.append(fit_ridge(normalized[i],target,self.lam))
        return [dict(state=packed[i],last=hx[i,-1].clone(),ridge=models[i]) for i in range(batch)]

    def advance(self,eyes,current):
        """Advance actual history, then read the same fixed observed-frame model."""
        torch=self.torch;cur=current.half().float()
        if self.arm=='R1':
            state,f=receptor_step(np.stack([e['state'] for e in eyes]),cur.cpu().numpy());packed=[x.copy() for x in state]
        else:
            state=self._unpack(torch.stack([e['state'] for e in eyes]));last=torch.stack([e['last'] for e in eyes]);state,f=self.bb.step(state,last,cur)
            f=f.double().cpu().numpy();p=self._pack(state);packed=[x.clone() for x in p]
        normalized=self.features(f);answers=[]
        for i,e in enumerate(eyes):
            e['state']=packed[i];e['last']=cur[i].clone();score=ridge_response(e['ridge'],normalized[i]);answers.append(map_center(score,self.geom))
        return answers

    def _new(self,tracks):
        from tracking.foveated_eyes import box_to_crop_frame
        histories=[]
        for t in tracks:
            h=list(self.hist[int(t.track_id)]);histories.append([h[0]]*(N_EXEMPLAR-len(h))+h[-N_EXEMPLAR:])
        hx=self.torch.stack([self.torch.stack([x[0] for x in h]) for h in histories])
        masks=np.stack([self.geom.mask_in_box(box_to_crop_frame(h[-1][1],h[-1][2],self.bank.eye_size)) for h in histories])
        eyes=self.start_sequence(hx,masks)
        for t,e in zip(tracks,eyes):self.eyes[int(t.track_id)]=e
        self.stats['births']+=len(tracks)

    def before_associate(self,pool,eye_frame,tracked_state):
        from tracking.foveated_eyes import crop_box
        self.frame_size=(eye_frame.shape[1],eye_frame.shape[0])
        if self.tau is None:return
        lost=[t for t in pool if t.state!=tracked_state and int(t.track_id) in self.hist];alive={int(t.track_id) for t in lost}
        for key in list(self.eyes):
            if key not in alive:self.eyes.pop(key)
        with self.torch.no_grad():
            new=[t for t in lost if int(t.track_id) not in self.eyes]
            for left in range(0,len(new),self.max_batch):self._new(new[left:left+self.max_batch])
            for left in range(0,len(lost),self.max_batch):
                group=lost[left:left+self.max_batch];crops=[crop_box(tuple(float(v) for v in t.mean[:4]),self.frame_size) for t in group]
                current=self.bank._render(eye_frame,crops)[:,0];answers=self.advance([self.eyes[int(t.track_id)] for t in group],current)
                for t,crop,(eye,confidence) in zip(group,crops,answers):
                    self.stats['steps']+=1;center=np.asarray(to_frame(eye,crop,self.bank.eye_size))
                    if not np.isfinite(center).all() or not np.isfinite(confidence):
                        self.stats['nonfinite_refusals']+=1;self.stats['refusals']+=1;continue
                    if confidence>=self.tau:
                        before=t.mean.copy();t.mean[:2]=center;assert np.array_equal(before[2:],t.mean[2:]);self.stats['interventions']+=1
                    else:self.stats['refusals']+=1
