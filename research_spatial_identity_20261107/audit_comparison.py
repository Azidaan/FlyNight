"""Independent correlation, certificate and choice audit of saved correspondences."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys
import csv
import gzip
import hashlib
import time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write,verified_protocol,guard,status,MSK,datetime

ARMS=['flyvis_spatial','flyvis_scrambled','receptor_spatial','r1_spatial']


def correlation(x,y):
    """Independent explicit centered-product formula, different reduction order."""
    x=x-x.sum(0)/len(x);y=y-y.sum(1,keepdims=True)/y.shape[1]
    xx=np.square(x).sum();yy=np.square(y).sum((1,2));out=np.full(len(y),np.nan)
    ok=(xx>=1e-12)&(yy>=1e-12)
    out[ok]=(y[ok]*x).sum((1,2))/np.sqrt(xx*yy[ok])
    return np.clip(out,-1,1)


def independent_geometry(rec,g):
    """Count size/offset and position grids independently, without map scores."""
    def grid(low,high,step,mean=None):
        if low>high:return []
        values=list(np.arange(low,high+1e-10,step));values.append(high)
        if mean is not None and low<=mean<=high:values.append(mean)
        return sorted(set(values))
    mu=np.array(rec['mean']);cv=np.array(rec['covariance']);box0=np.array(rec['history'][-1]['box'])
    span=box0[2:]-box0[:2];crop0=rec['history'][-1]['crop'];crop=rec['current_crop']
    p0=np.array(crop0[:2])+np.array(g['points'])*(np.array(crop0[2:])-crop0[:2])/g['eye_size']
    pt=np.array(crop[:2])+np.array(g['points'])*(np.array(crop[2:])-crop[:2])/g['eye_size']
    ref=((p0>=box0[:2])&(p0<=box0[2:])).all(1)
    past=g['spacing']*(crop0[2]-crop0[0])/g['eye_size']/2
    now=g['spacing']*(crop[2]-crop[0])/g['eye_size']/2
    results=[]
    for b in rec['candidate_boxes']:
        b=np.array(b);dims=[];valids=[];widths=[]
        for d in [0,1]:
            s=mu[d+2];sigma=np.sqrt(cv[d+2,d+2]);observed=b[d+2]-b[d]
            sizes=grid(max(observed,s-2*sigma),s+2*sigma,now,s);widths.append(len(sizes))
            count=0;inside=0
            for full in sizes:
                scale=full/span[d]
                for offset in grid(0,max(0,span[d]-observed/scale),past):
                    left=b[d]-scale*offset;right=left+full;count+=1
                    sd_left=np.sqrt(max(0,cv[d,d]+cv[d+2,d+2]/4-cv[d,d+2]))
                    sd_right=np.sqrt(max(0,cv[d,d]+cv[d+2,d+2]/4+cv[d,d+2]))
                    inside+=int(abs(left-(mu[d]-s/2))<=2*sd_left+1e-9 and abs(right-(mu[d]+s/2))<=2*sd_right+1e-9)
            dims.append(count);valids.append(inside)
        current=((pt>=b[:2])&(pt<=b[2:])).all(1)
        results.append(dict(all_transforms=dims[0]*dims[1],position_transforms=valids[0]*valids[1],
            current_nodes=int(current.sum()),reference_nodes=int(ref.sum()),width_values=widths[0],height_values=widths[1]))
    return results


def decision(scores,old,delta):
    if any(x.get('certified_no_dominance') for x in scores):return old
    if any(x['minimum'] is None or x['invalid_position'] for x in scores):return old
    for a,b,sign in [(0,1,1),(1,0,-1)]:
        if scores[a]['minimum']>0 and scores[a]['minimum']-scores[b]['maximum']>=delta and scores[a]['minimum']>scores[b]['all_maximum']:return sign
    return old


def main(obs):
    began=time.perf_counter();verified_protocol()
    protocol=read(RUN/'audit_phase_protocol.json')
    for name,h in protocol['sources'].items():assert sha(ROOT/name)==h,name
    rows=read(ROOT/'research_spatial_identity_20261107/readiness_rows.json')
    feature_manifest=read(RUN/'features_completed.json')['files']
    for name,h in feature_manifest.items():assert sha(RUN/'features'/name)==h,name
    norm=read(RUN/'image_normalization.json');scale=np.array(norm['sd']);scale[scale<1e-12]=1
    geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');tracks={}
    for video in {r['video'] for r in rows}:
        report=read(RUN/'tracker'/(video+'.json'));tracks.update({r['uid']:r for r in report['rows']})
    checked=0;max_error=0.;mappings=0;geometry_checks=0;details=[]
    for file in sorted((RUN/'scores').glob('*.json')):
        s=read(file);uid=s['uid'];rec=tracks[uid]
        actual_geo=independent_geometry(rec,geom)
        for actual,saved in zip(actual_geo,s['geometry']):
            for k,v in actual.items():assert v==saved[k],(uid,k,v,saved[k])
            geometry_checks+=1
        if s.get('lazy_geometry_rejection'):
            assert actual_geo[0]['reference_nodes']<3 or any(x['position_transforms']==0 or x['current_nodes']<3 for x in actual_geo)
        else:
            with np.load(RUN/'features'/f'{uid}.npz',allow_pickle=False) as z:
                fnow=z['fly_now'].astype(float);fref=z['fly_ref'].astype(float)
                inow=(z['image_now'].astype(float)-norm['mean'])/scale;iref=(z['image_ref'].astype(float)-norm['mean'])/scale
                rnow=z['r1_now'].astype(float);rref=z['r1_ref'].astype(float)
            extrema=[]
            for k,artifact in enumerate(s['mappings']):
                mp=ROOT/artifact['path'];assert sha(mp)==artifact['sha256']
                with np.load(mp,allow_pickle=False) as z:ptr=z['ptr'];ps=z['p'];qs=z['q'];inside=z['inside'];ids=z['ref_ids']
                shuffled=[]
                for rep in range(32):
                    seed=int.from_bytes(hashlib.sha256(f"{s['video']}|{rec['episode']}|{rep}|42".encode()).digest()[:8],'little')
                    array=fref.copy();array[ids]=fref[np.random.default_rng(seed).permutation(ids)];shuffled.append(array)
                shuffled=np.array(shuffled)
                low=np.full(4,np.inf);high=np.full(4,-np.inf);all_high=np.full(4,-np.inf);invalid=np.zeros(4,bool);unknown=np.zeros(4,bool)
                for j in range(len(inside)):
                    pp=ps[ptr[j]:ptr[j+1]];qq=qs[ptr[j]:ptr[j+1]];unique=np.unique(qq);v=np.full(4,np.nan)
                    if len(unique)>=3:
                        order=np.argsort(qq,kind='stable')
                        _,starts,counts=np.unique(qq[order],return_index=True,return_counts=True)
                        def average(x):return np.add.reduceat(x[pp[order]],starts,axis=0)/counts[:,None]
                        current=average(fnow)
                        v[0]=correlation(current,fref[unique][None])[0]
                        perm=correlation(current,shuffled[:,unique])
                        if np.isfinite(perm).all():v[1]=perm.mean()
                        v[2]=correlation(average(inow),iref[unique][None])[0]
                        v[3]=correlation(average(rnow),rref[unique][None])[0]
                    finite=np.isfinite(v);unknown|=~finite;all_high[finite]=np.maximum(all_high[finite],v[finite])
                    if inside[j]:invalid|=~finite;low[finite]=np.minimum(low[finite],v[finite]);high[finite]=np.maximum(high[finite],v[finite])
                    mappings+=1
                all_high[unknown]=1
                extrema.append((low,high,invalid))
                for i,arm in enumerate(ARMS):
                    saved=s['arms'][arm][k]
                    assert saved['invalid_position']==bool(invalid[i]) and saved['unknown_outside']==bool(unknown[i])
                    for key,value in [('minimum',low[i]),('maximum',high[i]),('all_maximum',all_high[i])]:
                        if not np.isfinite(value):assert saved[key] is None
                        else:
                            err=abs(saved[key]-value);max_error=max(max_error,err);assert err<1e-10,(uid,arm,key,err)
            for i,arm in enumerate(ARMS):
                cert=s.get('certificates',{}).get(arm)
                if cert=='undefined_position':assert extrema[0][2][i] or extrema[1][2][i]
                elif cert=='overlapping_subset_intervals':assert extrema[0][0][i]<=extrema[1][1][i] and extrema[1][0][i]<=extrema[0][1][i]
                elif cert=='nonpositive_minimum_witnesses':assert extrema[0][0][i]<=0 and extrema[1][0][i]<=0
                elif cert is not None:raise AssertionError(cert)
        checked+=1
        if checked%25==0:status('independent_score_audit',complete=checked,total=len(list((RUN/'scores').glob('*.json'))),guard=obs.read())
    evaluation=read(RUN/'evaluation_rows.json');selection=read(RUN/'selection.json');report=read(RUN/'comparison_report.json')
    for r in evaluation:
        scores=read(RUN/'scores'/f"{r['uid']}.json")
        for arm in ARMS:
            assert decision(scores['arms'][arm],r['old_choice'],selection['selected'][arm])==r['arms'][arm]['choice']
            # The reported zero effect is also true at every permitted threshold.
            for delta in [0,.05,.1,.2,.4,3]:assert decision(scores['arms'][arm],r['old_choice'],delta)==r['old_choice']
    assert sum(r['old_credit'] for r in evaluation)==report['baseline_credit']==107
    assert all(x['difference_pp']==0 and x['ci98_75_pp']==[0,0] for x in report['primary'].values())
    verified_protocol()
    write(RUN/'independent_audit.json',dict(passed=True,score_cases=checked,unique_correspondences=mappings,
        geometry_candidate_checks=geometry_checks,feature_files_verified=len(feature_manifest),max_correlation_error=max_error,
        evaluation_decisions_checked=len(evaluation)*4,all_thresholds_identical_to_baseline=True,
        seconds=time.perf_counter()-began,limitation='Independent score reductions, geometry counts, rejection certificates and decisions. Correspondence construction shares the saved mapping artifacts; it is not a second independent tracker implementation.'),new=True)


if __name__=='__main__':main(guard())
