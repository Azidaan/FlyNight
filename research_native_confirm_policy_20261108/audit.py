"""Independently check spatial arithmetic, optional assignments, labels and intervals."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_native_confirm_policy_20261108.common import *


def main(obs):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    phase('independent_audit_phase.json');phase('comparison_phase.json');report=read(RUN/'comparison_report.json');p=read(RUN/'protocol.json');assert sha(RUN/'selection.json')==report['selection_sha256'];inventory=read(RUN/'inventory_recovery1.json');features=read(RUN/'features_complete.json')['files'];norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');mean=np.asarray(norm['mean']);sd=np.where(np.asarray(norm['sd'])>=1e-12,norm['sd'],1);counts=Counter()
    grouped={}
    for e in inventory['events']:grouped.setdefault(e['video'],[]).append(e)
    for video,items in grouped.items():
        data=read(TRACE/'native_trace'/(video+'.json'));lookup={(d['frame'],e['old_tid'],e['new_tid']):e for d in data['events'] for e in d['edges']};scores=read(RUN/'scores'/(video+'.json'))['edges'];cache={}
        def get(key):
            if key not in cache:
                path=RUN/'features'/(key+'.npz');assert sha(path)==features[key]
                with np.load(path,allow_pickle=False) as z:cache[key]={k:z[k].astype(float) if k!='DIS_VALID' else z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE']-mean)/sd
            return cache[key]
        def correlation(x,y):
            x=x-x.mean(axis=0);y=y-y.mean(axis=0);xx=np.square(x).sum();yy=np.square(y).sum()
            return None if xx<1e-12 or yy<1e-12 else float(np.clip((x*y).sum()/np.sqrt(xx*yy),-1,1))
        for item in items:
            key=(item['frame'],item['old_tid'],item['new_tid']);e=lookup[key];cur=get(item['current_key']);ref=get(item['reference_key']);pp=np.asarray(e['transform']['p']);qq=np.asarray(e['transform']['q']);keep=cur['DIS_VALID'][pp]&ref['DIS_VALID'][qq];pp=pp[keep];qq=qq[keep];unique=np.unique(qq);assert len(unique)==e['support'];s=scores['|'.join(map(str,key))]['scores']
            assert all(f<=item['frame'] for f,crop in inventory['requests'][item['current_key']]['sequence']);assert max(f for f,c in inventory['requests'][item['reference_key']]['sequence'])<item['frame']
            for arm,name in [('IMAGE','IMAGE'),('R1','R1'),('DIS','DIS'),('FLYVIS','C')]:
                x=np.asarray([cur[name][pp[qq==q]].mean(axis=0) for q in unique]);y=ref[name][unique]
                if arm=='DIS':
                    x=x/np.asarray(e['transform']['scale']);den=np.sqrt(np.square(x).sum()*np.square(y).sum());value=None if np.square(x).sum()<1e-12 or np.square(y).sum()<1e-12 else float(np.clip((x*y).sum()/den,-1,1))
                else:value=correlation(x,y)
                assert (value is None and s[arm] is None) or value is not None and s[arm] is not None and abs(value-s[arm])<1e-10,(video,key,arm,value,s[arm]);counts['independent_scores']+=1
        for arm in report['results']['fit20']:
            path=RUN/'outcomes'/arm/(video+'.json');outcome=read(path);assert sha(path)==report['artifacts'][path.relative_to(ROOT).as_posix()];computed=Counter(N=outcome['totals']['N'])
            for d in outcome['decisions']:
                allowed=d['allowed'];links=d['links'];rr=sorted({e['old_tid'] for e in allowed});cc=sorted({e['new_tid'] for e in allowed});R=len(rr);C=len(cc)
                assert len({e['old_tid'] for e in links})==len(links)==len({e['new_tid'] for e in links})==len({e['canonical_id'] for e in links})
                if R and C:
                    mat=np.full((R+C,R+C),np.inf);mat[R:,C:]=0
                    for i in range(R):mat[i,C+i]=p['matching_threshold']/2
                    for j in range(C):mat[R+j,j]=p['matching_threshold']/2
                    for e in allowed:
                        assert abs(e['corrected_cost']-(e['cost']-p['alpha']))<1e-12;mat[rr.index(e['old_tid']),cc.index(e['new_tid'])]=np.float32(e['corrected_cost'])
                    r,c=linear_sum_assignment(mat);optimal=float(mat[r,c].sum());actual=sum(float(np.float32(e['corrected_cost'])) for e in links)+(R+C-2*len(links))*p['matching_threshold']/2
                    assert abs(actual-optimal)<1e-5,(video,arm,d['frame'],actual,optimal);counts['independent_optional_assignments']+=1
                computed['revoked']+=len(d['revoked'])
            for e in outcome['links']:
                kind=e['kind'];computed['links']+=1;computed[kind]+=1;computed['direct_current_known']+=bool(e['label'].get('direct_current_known'));label=e['label'];assert (kind=='unknown')== (not label['known'])
                if label['known']:assert (kind=='correct')==(label['identity']==label['target'])
            computed['J']=computed['correct']-computed['foreign'];assert dict(computed)==outcome['totals'];counts['outcome_ledgers']+=1
        status('native_confirmation_independent_audit',video=video,complete=counts['outcome_ledgers'],scores=counts['independent_scores'],guard=obs.read())
    videos=sorted(report['results']['evaluate26']['B0']['per_video']);ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    for key,value in report['comparisons'].items():
        a,b=key.split('-');A=report['results']['evaluate26'][a]['per_video'];B=report['results']['evaluate26'][b]['per_video'];d=np.array([A[v].get('J',0)-B[v].get('J',0) for v in videos]);n=np.array([A[v]['N'] for v in videos]);sample=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1));assert value['delta_J']==int(d.sum());assert np.allclose(value['ci95'],np.percentile(sample,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(sample,[2.5/12,100-2.5/12]),atol=1e-12,rtol=0);counts['bootstrap_intervals']+=1
    for n,h in report['label_sources'].items():assert sha(ROOT/n)==h
    phase('independent_audit_phase.json');phase('comparison_phase.json');write(RUN/'independent_audit.json',dict(passed=True,counts=dict(counts),comparison_sha256=sha(RUN/'comparison_report.json'),scope='Independent map arithmetic and SciPy augmented optional assignment objective, outcome totals and bootstrap. Native full physical parity separately checked before this study. No full tracker benefit inferred.'),new=True)


if __name__=='__main__':main(guard())
