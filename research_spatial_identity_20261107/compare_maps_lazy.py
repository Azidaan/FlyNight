"""Exact lazy evaluation of the frozen robust-dominance decision.

Partial score intervals are identified explicitly. A rejection certificate is
sufficient for the same decision over the entire originally specified grid.
"""
import sys
import time
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_spatial_identity_20261107.compare_maps as base
import numpy as np
from scipy.spatial import cKDTree
from research_spatial_identity_20261107.run_common import RUN,read,write,sha,guard


def geometry(rec,geom):
    points=np.asarray(geom['points'],float);size=geom['eye_size']
    c0=np.array(rec['history'][-1]['crop']);ct=np.array(rec['current_crop'])
    b0=np.array(rec['history'][-1]['box']);bsize=b0[2:]-b0[:2]
    p0=base.prepare_points(points,c0,size);pt=base.prepare_points(points,ct,size)
    refmask=((p0>=b0[:2])&(p0<=b0[2:])).all(1)
    sp0=geom['spacing']*(c0[2]-c0[0])/size;spt=geom['spacing']*(ct[2]-ct[0])/size
    mu=np.array(rec['mean']);cv=np.array(rec['covariance']);sd=np.sqrt(np.diag(cv))
    pred=np.r_[mu[:2]-mu[2:4]/2,mu[:2]+mu[2:4]/2]
    edges=np.sqrt(np.maximum([cv[0,0]+cv[2,2]/4-cv[0,2],cv[1,1]+cv[3,3]/4-cv[1,3],
                              cv[0,0]+cv[2,2]/4+cv[0,2],cv[1,1]+cv[3,3]/4+cv[1,3]],0))
    candidates=[]
    for box in rec['candidate_boxes']:
        box=np.array(box);bs=box[2:]-box[:2];components=[];dims=[]
        for d in range(2):
            sizes=base.axis_grid(max(bs[d],mu[d+2]-2*sd[d+2]),mu[d+2]+2*sd[d+2],spt/2,mu[d+2])
            dims.append(len(sizes));axis=[]
            for full in sizes:
                gamma=full/bsize[d]
                for offset in base.axis_grid(0,max(0,bsize[d]-bs[d]/gamma),sp0/2):
                    lo=box[d]-gamma*offset
                    inside=abs(lo-pred[d])<=2*edges[d]+1e-9 and abs(lo+full-pred[d+2])<=2*edges[d+2]+1e-9
                    axis.append((gamma,offset,bool(inside)))
            components.append(axis)
        indices=np.flatnonzero(((pt>=box[:2])&(pt<=box[2:])).all(1))
        count_inside=sum(x[2] for x in components[0])*sum(x[2] for x in components[1])
        candidates.append(dict(box=box,indices=indices,axes=components,stats=dict(
            all_transforms=len(components[0])*len(components[1]),position_transforms=count_inside,
            current_nodes=len(indices),reference_nodes=int(refmask.sum()),width_values=dims[0],height_values=dims[1])))
    return b0,p0,pt,refmask,sp0,candidates


def transforms(candidate):
    # Position-admissible transforms first. Ordering cannot change a robust decision.
    axes=candidate['axes']
    for phase in [True,False]:
        for gx,tx,ix in axes[0]:
            for gy,ty,iy in axes[1]:
                inside=ix and iy
                if inside==phase:yield gx,gy,tx,ty,inside


def choose(scores,old,delta):
    if any(x.get('certified_no_dominance') for x in scores):
        return old,'certified_no_dominance'
    return base_original_choose(scores,old,delta)


base_original_choose=base.choose


def score_case(r,rec,geom,norm):
    began=time.perf_counter();uid=r['uid'];path=RUN/'features'/f'{uid}.npz'
    b0,p0,pt,refmask,spacing,candidates=geometry(rec,geom)
    blank=lambda:dict(minimum=None,maximum=None,all_maximum=None,invalid_position=False,unknown_outside=False)
    # This geometric condition is sufficient for every arm to retain the old decision.
    if refmask.sum()<3 or any(c['stats']['position_transforms']==0 or len(c['indices'])<3 for c in candidates):
        return dict(uid=uid,video=r['video'],split=r['split'],arms={a:[blank(),blank()] for a in base.ARMS},
            geometry=[c['stats'] for c in candidates],mappings=[],lazy_geometry_rejection=True,
            partial_intervals=True,seconds=time.perf_counter()-began,features_sha256=sha(path))
    with np.load(path,allow_pickle=False) as z:
        fnow=z['fly_now'].astype(float);fref=z['fly_ref'].astype(float)
        inow=z['image_now'].astype(float);iref=z['image_ref'].astype(float)
        rnow=z['r1_now'].astype(float);rref=z['r1_ref'].astype(float)
    scale=np.array(norm['sd']);scale=np.where(scale>=1e-12,scale,1)
    inow=(inow-norm['mean'])/scale;iref=(iref-norm['mean'])/scale
    ids=np.flatnonzero(refmask);permuted=[]
    for j in range(32):
        seed=int.from_bytes(hashlib.sha256(f"{r['video']}|{r['episode']}|{j}|42".encode()).digest()[:8],'little')
        copy=fref.copy();copy[ids]=fref[np.random.default_rng(seed).permutation(ids)];permuted.append(copy)
    shuffled=np.stack(permuted);tree=cKDTree(p0)
    lo=np.full((2,4),np.inf);hi=np.full((2,4),-np.inf);allhi=np.full((2,4),-np.inf)
    invalid=np.zeros((2,4),bool);unknown=np.zeros((2,4),bool)
    iterators=[iter(transforms(c)) for c in candidates];exhausted=[False,False];caches=[{},{}]
    evaluated=[0,0];certificates={}
    while not all(exhausted):
        for k,c in enumerate(candidates):
            if exhausted[k]:continue
            try:gx,gy,tx,ty,inside=next(iterators[k])
            except StopIteration:exhausted[k]=True;continue
            evaluated[k]+=1
            projected=b0[:2]+[tx,ty]+(pt[c['indices']]-c['box'][:2])/[gx,gy]
            dist,q=tree.query(projected)
            valid=((projected>=b0[:2])&(projected<=b0[2:])).all(1)&refmask[q]&(dist<=.6*spacing)
            pp=c['indices'][valid];qq=q[valid]
            key=np.column_stack([pp,qq]).astype(np.int16).tobytes()
            if key not in caches[k]:
                uq,inv=np.unique(qq,return_inverse=True);scores=np.full(4,np.nan)
                if len(uq)>=3:
                    counts=np.bincount(inv)[:,None]
                    def aggregate(x):
                        a=np.zeros((len(uq),x.shape[1]));np.add.at(a,inv,x[pp]);return a/counts
                    fc=aggregate(fnow);scores[0]=base.centered_correlation(fc,fref[uq][None])[0]
                    ss=base.centered_correlation(fc,shuffled[:,uq])
                    if np.isfinite(ss).all():scores[1]=ss.mean()
                    scores[2]=base.centered_correlation(aggregate(inow),iref[uq][None])[0]
                    scores[3]=base.centered_correlation(aggregate(rnow),rref[uq][None])[0]
                caches[k][key]=[scores,False,pp,qq]
            entry=caches[k][key];entry[1]|=inside;scores=entry[0];finite=np.isfinite(scores)
            allhi[k,finite]=np.maximum(allhi[k,finite],scores[finite]);unknown[k]|=~finite
            if inside:
                invalid[k]|=~finite;lo[k,finite]=np.minimum(lo[k,finite],scores[finite]);hi[k,finite]=np.maximum(hi[k,finite],scores[finite])
        for i,arm in enumerate(base.ARMS):
            if invalid[:,i].any():certificates[arm]='undefined_position'
            elif (lo[0,i]<=hi[1,i] and lo[1,i]<=hi[0,i]):certificates[arm]='overlapping_subset_intervals'
            elif (lo[:,i]<=0).all():certificates[arm]='nonpositive_minimum_witnesses'
        if len(certificates)==len(base.ARMS):break
    allhi[unknown]=1
    result={}
    for i,arm in enumerate(base.ARMS):
        result[arm]=[]
        for k in range(2):
            row=dict(minimum=float(lo[k,i]) if np.isfinite(lo[k,i]) else None,
                maximum=float(hi[k,i]) if np.isfinite(hi[k,i]) else None,
                all_maximum=float(allhi[k,i]) if np.isfinite(allhi[k,i]) else None,
                invalid_position=bool(invalid[k,i]),unknown_outside=bool(unknown[k,i]))
            if arm in certificates:row['certified_no_dominance']=certificates[arm]
            result[arm].append(row)
    artifacts=[]
    for k,cache in enumerate(caches):
        mp=RUN/'geometry'/f'{uid}_{k}_lazy.npz';mp.parent.mkdir(exist_ok=True)
        ptr=np.r_[0,np.cumsum([len(v[2]) for v in cache.values()])]
        pp=np.concatenate([v[2] for v in cache.values()]) if cache else np.empty(0,int)
        qq=np.concatenate([v[3] for v in cache.values()]) if cache else np.empty(0,int)
        np.savez_compressed(mp,ptr=ptr,p=pp.astype(np.int16),q=qq.astype(np.int16),
            inside=np.array([v[1] for v in cache.values()],bool),ref_ids=ids)
        artifacts.append(dict(path=mp.relative_to(ROOT).as_posix(),sha256=sha(mp)))
        candidates[k]['stats'].update(evaluated_transforms=evaluated[k],unique_mappings=len(cache))
    return dict(uid=uid,video=r['video'],split=r['split'],arms=result,
        geometry=[c['stats'] for c in candidates],mappings=artifacts,certificates=certificates,
        partial_intervals=not all(exhausted),seconds=time.perf_counter()-began,features_sha256=sha(path))


if __name__=='__main__':
    observer=guard()
    phase=read(RUN/'lazy_phase_protocol.json')
    for name,h in phase['source_sha256'].items():assert sha(ROOT/name)==h,name
    base.score_case=score_case;base.choose=choose
    base.main(observer)
