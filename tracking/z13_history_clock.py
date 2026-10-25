"""Advance the frozen retina/neural state by actual stored video-frame intervals."""
import numpy as np


def frame_intervals(frames,shape):
    frames=np.asarray(frames)
    if frames.shape!=shape or frames.dtype.kind not in 'iu':raise ValueError('Aligned integer frame indices required')
    delta=np.diff(frames,axis=1)
    if np.any(delta<0):raise ValueError('Frame order must be nondecreasing within an episode')
    return delta


def clock_features(torch,arm,bb,hexes,neighbors,needed,frames):
    from scripts.z13_target_model import neighbor_aug,R1_TAUS
    delta=frame_intervals(frames,hexes.shape[:2]);out={};last=hexes[:,0]
    if arm=='R1':
        values=hexes.cpu().numpy().astype(np.float64);ema=np.repeat(values[:,:1],len(R1_TAUS),axis=1);a=(1/R1_TAUS)[None,:,None]
        for t in range(1,values.shape[1]):
            count=delta[:,t-1]
            for k in range(1,int(count.max(initial=0))+1):
                fraction=np.minimum(k/np.maximum(count,1),1.)[:,None]
                current=values[:,t-1]*(1-fraction)+values[:,t]*fraction
                updated=ema+(current[:,None]-ema)*a
                ema=np.where((count>=k)[:,None,None],updated,ema)
            if t in needed:out[t]=neighbor_aug(np.concatenate([ema.copy(),values[:,t,None]-ema],axis=1),neighbors)
        return out
    with torch.no_grad():
        state=bb.init(hexes[:,0:1])
        for t in range(1,hexes.shape[1]):
            count=delta[:,t-1];previous=last
            for k in range(1,int(count.max(initial=0))+1):
                fraction=torch.as_tensor(np.minimum(k/np.maximum(count,1),1.),device=hexes.device,dtype=hexes.dtype)[:,None]
                current=last*(1-fraction)+hexes[:,t]*fraction
                next_state,_=bb.step(state,previous,current)
                active=torch.as_tensor(count>=k,device=hexes.device)
                if not bool(active.all()):
                    for group in ('nodes','edges'):
                        old=getattr(state,group);new=getattr(next_state,group)
                        for key in old:
                            shape=(len(active),)+(1,)*(old[key].ndim-1)
                            new[key]=torch.where(active.reshape(shape),new[key],old[key])
                    next_state=bb.net._state_api(next_state)
                state=next_state;previous=current
            last=hexes[:,t]
            if t in needed:out[t]=neighbor_aug(state.nodes.activity[:,bb.index].double().cpu().numpy(),neighbors)
    return out
