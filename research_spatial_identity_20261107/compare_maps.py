"""Finite, shared geometry and frozen spatial-map comparison on archived pairs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import csv
import gzip
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write,status,verified_protocol,guard

ARMS=('flyvis_spatial','flyvis_scrambled','receptor_spatial','r1_spatial')


def axis_grid(lo,hi,step,extra=None):
    if hi<lo:return np.empty(0)
    x=np.r_[np.arange(lo,hi+1e-10,step),hi]
    if extra is not None and lo<=extra<=hi:x=np.r_[x,extra]
    return np.unique(x)


def centered_correlation(current,reference):
    """reference is (replicates,nodes,channels); constant profiles are undefined."""
    x=current-current.mean(0,keepdims=True)
    y=reference-reference.mean(1,keepdims=True)
    vx=float(np.sum(x*x));vy=np.sum(y*y,axis=(1,2))
    values=np.full(len(reference),np.nan)
    valid=(vx>=1e-12)&(vy>=1e-12)
    values[valid]=np.einsum('rnc,nc->r',y[valid],x)/np.sqrt(vx*vy[valid])
    return np.clip(values,-1,1)


def prepare_points(points,crop,eye_size):
    crop=np.asarray(crop,float)
    return crop[:2]+points*(crop[2:]-crop[:2])/eye_size


def score_case(r,rec,geom,image_norm):
    began=time.perf_counter(); uid=r['uid']
    feature_path=RUN/'features'/(str(uid)+'.npz')
    with np.load(feature_path,allow_pickle=False) as z:
        assert int(z['uid'])==uid
        fly_now=z['fly_now'].astype(float);fly_ref=z['fly_ref'].astype(float)
        r1_now=z['r1_now'].astype(float);r1_ref=z['r1_ref'].astype(float)
        im_now=z['image_now'].astype(float);im_ref=z['image_ref'].astype(float)
    mean=np.asarray(image_norm['mean']);scale=np.asarray(image_norm['sd']);scale=np.where(scale>=1e-12,scale,1)
    im_now=(im_now-mean)/scale;im_ref=(im_ref-mean)/scale
    points=np.asarray(geom['points'],float);size=geom['eye_size'];spacing=geom['spacing']
    c0=np.asarray(rec['history'][-1]['crop'],float);ct=np.asarray(rec['current_crop'],float)
    b0=np.asarray(rec['history'][-1]['box'],float);b0size=b0[2:]-b0[:2]
    p0=prepare_points(points,c0,size);pt=prepare_points(points,ct,size)
    refmask=((p0>=b0[:2])&(p0<=b0[2:])).all(1)
    ref_ids=np.flatnonzero(refmask)
    tree=cKDTree(p0)
    past_spacing=spacing*(c0[2]-c0[0])/size
    current_spacing=spacing*(ct[2]-ct[0])/size
    permutations=[]
    for rep in range(32):
        token=f"{r['video']}|{r['episode']}|{rep}|42".encode()
        seed=int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
        a=fly_ref.copy();a[ref_ids]=fly_ref[np.random.default_rng(seed).permutation(ref_ids)];permutations.append(a)
    shuffled=np.stack(permutations)
    native=np.asarray(rec['mean']);cov=np.asarray(rec['covariance']);cx,cy,ww,hh=native[:4]
    pred=np.array([cx-ww/2,cy-hh/2,cx+ww/2,cy+hh/2])
    edge_sd=np.sqrt(np.maximum([cov[0,0]+cov[2,2]/4-cov[0,2],cov[1,1]+cov[3,3]/4-cov[1,3],
                                cov[0,0]+cov[2,2]/4+cov[0,2],cov[1,1]+cov[3,3]/4+cov[1,3]],0))
    size_sd=np.sqrt(np.maximum([cov[2,2],cov[3,3]],0))
    results={a:[] for a in ARMS}; geometry_stats=[]
    mapping_artifacts=[]
    for candidate_number,box in enumerate(rec['candidate_boxes']):
        box=np.asarray(box,float);bs=box[2:]-box[:2]
        indices=np.flatnonzero(((pt>=box[:2])&(pt<=box[2:])).all(1))
        widths=axis_grid(max(bs[0],ww-2*size_sd[0]),ww+2*size_sd[0],current_spacing/2,ww)
        heights=axis_grid(max(bs[1],hh-2*size_sd[1]),hh+2*size_sd[1],current_spacing/2,hh)
        minimum=np.full(4,np.inf);maximum=np.full(4,-np.inf);all_max=np.full(4,-np.inf)
        invalid=np.zeros(4,bool);all_unknown=np.zeros(4,bool)
        counts=Counter();cache={}
        if len(ref_ids)>=3 and len(indices)>=3 and min(b0size)>0:
            for w in widths:
                gx=w/b0size[0]
                txs=axis_grid(0,max(0,b0size[0]-bs[0]/gx),past_spacing/2)
                for h in heights:
                    gy=h/b0size[1]
                    tys=axis_grid(0,max(0,b0size[1]-bs[1]/gy),past_spacing/2)
                    for tx in txs:
                        for ty in tys:
                            counts['all_transforms']+=1
                            implied_lo=box[:2]-np.array([gx*tx,gy*ty])
                            implied=np.r_[implied_lo,implied_lo+[w,h]]
                            inside=bool((np.abs(implied-pred)<=2*edge_sd+1e-9).all())
                            counts['position_transforms']+=int(inside)
                            projected=b0[:2]+[tx,ty]+(pt[indices]-box[:2])/[gx,gy]
                            dist,q=tree.query(projected)
                            valid=((projected>=b0[:2])&(projected<=b0[2:])).all(1)&refmask[q]&(dist<=.6*past_spacing)
                            pp=indices[valid];qq=q[valid]
                            signature=np.column_stack([pp,qq]).astype(np.int16).tobytes()
                            if signature not in cache:
                                unique,inverse=np.unique(qq,return_inverse=True)
                                scores=np.full(4,np.nan)
                                if len(unique)>=3:
                                    cnt=np.bincount(inverse).astype(float)[:,None]
                                    def aggregate(x):
                                        summed=np.zeros((len(unique),x.shape[1]))
                                        np.add.at(summed,inverse,x[pp]);return summed/cnt
                                    fc=aggregate(fly_now)
                                    scores[0]=centered_correlation(fc,fly_ref[unique][None])[0]
                                    shuffled_scores=centered_correlation(fc,shuffled[:,unique])
                                    if np.isfinite(shuffled_scores).all():scores[1]=shuffled_scores.mean()
                                    scores[2]=centered_correlation(aggregate(im_now),im_ref[unique][None])[0]
                                    scores[3]=centered_correlation(aggregate(r1_now),r1_ref[unique][None])[0]
                                cache[signature]=[scores,len(unique),False,pp,qq]
                            item=cache[signature]
                            item[2]|=inside
                            scores,n_unique=item[:2]
                            counts['insufficient_unique_nodes']+=int(n_unique<3)
                            finite=np.isfinite(scores)
                            all_max[finite]=np.maximum(all_max[finite],scores[finite])
                            all_unknown|=~finite
                            if inside:
                                invalid|=~finite
                                minimum[finite]=np.minimum(minimum[finite],scores[finite])
                                maximum[finite]=np.maximum(maximum[finite],scores[finite])
        # Undefined outside-position correlations get the conservative upper bound1.
        all_max[all_unknown]=1.0
        for i,arm in enumerate(ARMS):
            vals=[minimum[i],maximum[i],all_max[i]]
            results[arm].append(dict(minimum=float(vals[0]) if np.isfinite(vals[0]) else None,
                maximum=float(vals[1]) if np.isfinite(vals[1]) else None,
                all_maximum=float(vals[2]) if np.isfinite(vals[2]) else None,
                invalid_position=bool(invalid[i]),unknown_outside=bool(all_unknown[i])))
        geometry_stats.append(dict(counts,unique_mappings=len(cache),current_nodes=len(indices),reference_nodes=len(ref_ids),
                                   width_values=len(widths),height_values=len(heights)))
        # Preserve every distinct correspondence, including undefined support.
        # An independent evaluator can recompute every score, not just its extrema.
        mapping_path=RUN/'geometry'/f'{uid}_{candidate_number}.npz'
        mapping_path.parent.mkdir(exist_ok=True)
        ptr=np.r_[0,np.cumsum([len(x[3]) for x in cache.values()])]
        p=np.concatenate([x[3] for x in cache.values()]) if cache else np.empty(0,int)
        q=np.concatenate([x[4] for x in cache.values()]) if cache else np.empty(0,int)
        np.savez_compressed(mapping_path,ptr=ptr,p=p.astype(np.int16),q=q.astype(np.int16),
                            inside=np.array([x[2] for x in cache.values()],bool),ref_ids=ref_ids)
        mapping_artifacts.append(dict(path=mapping_path.relative_to(ROOT).as_posix(),sha256=sha(mapping_path)))
    return dict(uid=uid,video=r['video'],split=r['split'],arms=results,geometry=geometry_stats,
                mappings=mapping_artifacts,seconds=time.perf_counter()-began,features_sha256=sha(feature_path))


def choose(scores,old_choice,delta):
    if any(x['minimum'] is None or x['invalid_position'] for x in scores):return old_choice,'geometry_or_signal_undefined'
    for k,j in [(0,1),(1,0)]:
        if scores[k]['minimum']>0 and scores[k]['minimum']-scores[j]['maximum']>=delta and scores[j]['all_maximum']<scores[k]['minimum']:
            return (1 if k==0 else -1),'intervention'
    return old_choice,'no_robust_dominance'


def credit(choice,truth):
    return .5 if choice==0 else float(choice==truth)


def main(obs):
    verified_protocol();assert read(RUN/'features_completed.json')['passed']
    phase=read(RUN/'comparison_phase_protocol.json')
    for name,h in phase['source_sha256'].items():assert sha(ROOT/name)==h,name
    rows=read(ROOT/'research_spatial_identity_20261107/readiness_rows.json')
    geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');norm=read(RUN/'image_normalization.json')
    tracked={}
    for video in sorted({r['video'] for r in rows}):
        tracked.update({r['uid']:r for r in read(RUN/'tracker'/(video+'.json'))['rows']})
    folder=RUN/'scores';folder.mkdir(exist_ok=True)
    def run_rows(cases):
        output=[]
        for i,r in enumerate(cases):
            path=folder/(str(r['uid'])+'.json')
            if path.exists():value=read(path);assert value['features_sha256']==sha(RUN/'features'/(str(r['uid'])+'.npz'))
            else:value=score_case(r,tracked[r['uid']],geom,norm);write(path,value,new=True)
            output.append(value)
            status('spatial_comparison',split=r['split'],complete=i+1,total=len(cases),uid=r['uid'],case_seconds=value['seconds'],guard=obs.read())
            if i%10==0:print('score',r['split'],i+1,len(cases),r['uid'],round(value['seconds'],3),flush=True)
        return output
    first=min((r for r in rows if r['split']=='fit84'),key=lambda r:(r['video'],r['episode'],r['uid']))
    first_scores=run_rows([first])[0]
    write(RUN/'geometry_price.json',dict(uid=first['uid'],seconds=first_scores['seconds'],geometry=first_scores['geometry'],
        estimated269_cases_seconds=269*first_scores['seconds'],scope='one fixed fit84 case; scene complexity varies'),new=not (RUN/'geometry_price.json').exists())
    fit20=run_rows([r for r in rows if r['split']=='fit20'])
    # Evaluation labels are not loaded until the selected thresholds have been saved.
    selection_path=RUN/'selection.json'
    with gzip.open(ROOT/'runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz','rt',encoding='ascii') as stream:
        dev={int(r['uid']):r for r in csv.DictReader(stream) if r['split']=='fit20' and r['exactly2']=='1' and r['label_reason']=='ok'}
    if not selection_path.exists():
        selected={};grid={}
        for arm in ARMS:
            entries=[]
            for delta in [0,.05,.1,.2,.4,3.0]:
                total=0
                for value in fit20:
                    label=dev[value['uid']];ch,_=choose(value['arms'][arm],int(label['whole_choice']),delta)
                    total+=credit(ch,int(label['target_side']))
                entries.append(dict(delta=delta,credit=total))
            selected[arm]=max(entries,key=lambda x:(x['credit'],x['delta']))['delta'];grid[arm]=entries
        write(selection_path,dict(selected=selected,grid=grid,fit20_uids=[r['uid'] for r in fit20],
              source_sha256=sha(__file__)),new=True)
    selected=read(selection_path)['selected']
    evaluation=run_rows([r for r in rows if r['split']=='evaluate26'])
    with gzip.open(ROOT/'runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz','rt',encoding='ascii') as stream:
        labels={int(r['uid']):r for r in csv.DictReader(stream) if r['split']=='evaluate26' and r['exactly2']=='1' and r['label_reason']=='ok'}
    results=[]
    for value in evaluation:
        label=labels[value['uid']];old=int(label['whole_choice']);truth=int(label['target_side'])
        record=dict(uid=value['uid'],video=value['video'],truth=truth,old_choice=old,old_credit=credit(old,truth),arms={})
        for arm in ARMS:
            ch,reason=choose(value['arms'][arm],old,selected[arm])
            record['arms'][arm]=dict(choice=ch,credit=credit(ch,truth),reason=reason)
        results.append(record)
    clusters=read(ROOT/'runs/flyvis_exclusive_t4t5_20261107/protocol.json')['videos']['evaluate26']
    counts=np.array([sum(r['video']==v for r in results) for v in clusters],float)
    sampled=np.random.default_rng(42).integers(0,len(clusters),size=(20000,len(clusters)))
    denominator=counts[sampled].sum(1);valid=denominator>0
    def values(arm):return np.array([sum((r['old_credit'] if arm=='old_readout' else r['arms'][arm]['credit']) for r in results if r['video']==v) for v in clusters])
    primary={}
    for comparator in ['old_readout','flyvis_scrambled','receptor_spatial','r1_spatial']:
        diff=values('flyvis_spatial')-values(comparator)
        bootstrap=100*diff[sampled].sum(1)[valid]/denominator[valid]
        primary[comparator]=dict(difference_pp=float(100*diff.sum()/len(results)),
            ci95_pp=np.percentile(bootstrap,[2.5,97.5]).tolist(),ci98_75_pp=np.percentile(bootstrap,[.625,99.375]).tolist())
    summary={}
    for arm in ARMS:
        summary[arm]=dict(correct_credit=sum(r['arms'][arm]['credit'] for r in results),
            fixes=sum(r['old_credit']==0 and r['arms'][arm]['credit']==1 for r in results),
            breaks=sum(r['old_credit']==1 and r['arms'][arm]['credit']==0 for r in results),
            reasons=dict(Counter(r['arms'][arm]['reason'] for r in results)),delta=selected[arm])
    write(RUN/'evaluation_rows.json',results,new=True)
    report=dict(complete=True,pairs=len(results),clusters=len(clusters),baseline_credit=sum(r['old_credit'] for r in results),
        summary=summary,primary=primary,proceed=all(x['ci98_75_pp'][0]>0 for x in primary.values()),
        selection_sha256=sha(selection_path),rows_sha256=sha(RUN/'evaluation_rows.json'),
        limitation='Historical known-target pair diagnostic, not full-tracker benefit or an independent final test.')
    verified_protocol();write(RUN/'comparison_report.json',report,new=True)
    print(str(report),flush=True)


if __name__=='__main__':main(guard())
