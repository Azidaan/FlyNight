"""Separate joint14 explanation; do not rewrite the original11-comparison study."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_age_20261108.common import *


def main(obs):
    import numpy as np
    from research_identity_assignment_run_20261108.full_statistics import pooled
    from trackeval.metrics.hota import HOTA
    from evaluation.hota_eval import VALID
    phase('full_phase.json');protocol=read(RUN/'protocol.json');p=read(FULL/'protocol.json');old=read(OWN_FULL/'comparison_report.json');videos=list(p['videos']);n=len(videos)
    with np.load(OWN_FULL/'bootstrap_counts.npz',allow_pickle=False) as z:counts=z['counts']
    indices=np.random.default_rng(42).integers(0,n,(20000,n));assert np.array_equal(counts,np.array([np.bincount(i,minlength=n) for i in indices]))
    points={};samples={};per_video={};extra=read(FULL/'additional_report.json');components={}
    for arm in p['parameters']:
        folder=FULL if arm=='AGE' else OWN_FULL;data=[read(folder/'metrics'/arm/(v+'.json')) for v in videos]
        assert all(m['passed'] for m in data);c={k:np.array([m['components'][k] for m in data],float) for k in ['TP','FP','FN','AA']};components[arm]=c
        point={k:float(v[0]) for k,v in pooled(c,np.ones((1,n))).items()};identity={k:sum(m['components'][k] for m in data) for k in ['IDTP','IDFP','IDFN']}
        point['IDF1']=100*identity['IDTP']/max(1.,identity['IDTP']+.5*identity['IDFP']+.5*identity['IDFN'])
        for k in ['AssA','IDF1']:point['mean_video_'+k]=float(np.mean([m['metrics'][k] for m in data]))
        for k in ['IDSw','CLR_FP','CLR_FN']:point[k]=int(sum(m['metrics'][k] for m in data))
        point.update(extra['arms'][arm]['totals']);points[arm]=point;samples[arm]=pooled(c,counts)['HOTA'];per_video[arm]={v:m['metrics'] for v,m in zip(videos,data)}
        if arm!='AGE':assert point==old['pooled'][arm]
    comparisons={}
    for a,b in read(OWN_FULL/'protocol.json')['primary_comparisons']+protocol['full_comparisons']:
        key=a+'-'+b
        if key in comparisons:continue
        delta=samples[a]-samples[b];A=points[a];B=points[b]
        harm=dict(AssA_mean=A['mean_video_AssA']-B['mean_video_AssA'],IDF1_mean=A['mean_video_IDF1']-B['mean_video_IDF1'],
            IDSw=A['IDSw']-B['IDSw'],FP=A['CLR_FP']-B['CLR_FP'],FN=A['CLR_FN']-B['CLR_FN'],wrong_known_returns=A.get('wrong_returns',0)-B.get('wrong_returns',0))
        safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns'])
        comparisons[key]=dict(delta_pooled_HOTA=A['HOTA']-B['HOTA'],ci95=np.percentile(delta,[2.5,97.5]).tolist(),ci_joint14=np.percentile(delta,[2.5/14,100-2.5/14]).tolist(),harm=harm,harm_passed=safe)
    accepted={k:bool(v['ci_joint14'][0]>0 and v['harm_passed']) for k,v in comparisons.items()}
    gains={a:dict(correct_returns=points[a].get('correct_returns',0)-points['B0'].get('correct_returns',0),sustained_success=points[a].get('sustained_success',0)-points['B0'].get('sustained_success',0)) for a in ['AGE','FLYVIS','IMAGE']}
    closures={a:bool(accepted.get(a+'-B0',False) and gains[a]['correct_returns']>0 and gains[a]['sustained_success']>0) for a in gains}
    detail=[];official=HOTA();checks=0;maximum=0
    for video in videos:
        with gzip.open(FULL/'metrics/AGE'/(video+'.details.json.gz'),'rt',encoding='utf-8') as f:d=json.load(f)[video]
        detail.append({c:{k:np.array(v) if isinstance(v,list) else v for k,v in d[c]['HOTA'].items()} for c in VALID if c in d})
    for index,selected in enumerate([np.arange(n),*indices[:20]]):
        cls={}
        for category in VALID:
            seq={str(i):detail[int(j)][category] for i,j in enumerate(selected) if category in detail[int(j)]}
            if seq:cls[category]=official.combine_sequences(seq)
        value=float(official.combine_classes_det_averaged(cls)['HOTA'].mean()*100);expected=points['AGE']['HOTA'] if index==0 else samples['AGE'][index-1]
        error=abs(value-expected);assert error<1e-9;maximum=max(maximum,error);checks+=1
    with (FULL/'bootstrap_counts.npz').open('xb') as f:np.savez_compressed(f,counts=counts)
    phase('full_phase.json');write(FULL/'comparison_report.json',dict(passed=True,videos=videos,parameters=p['parameters'],pooled=points,per_video=per_video,comparisons=comparisons,accepted=accepted,
        identity_gain=gains,problem1_acceptance=closures,accepted_flyvis_system_and_increment=all(accepted['FLYVIS-'+a] for a in p['parameters'] if a!='FLYVIS'),
        bootstrap=dict(family=14,samples=20000,seed=42,primary_ci_percent=100*(1-.05/14),official_AGE_pooling_checks=checks,max_error=maximum),
        original_study_unchanged_sha256=sha(OWN_FULL/'comparison_report.json'),historical=True,independent_final_test=False,scope=protocol['limitation']),new=True)
    write(FULL/'analyses_complete.json',dict(passed=True,comparison_sha256=sha(FULL/'comparison_report.json')),new=True)


if __name__=='__main__':main(guard())
