"""Official pooled HOTA,11registered comparisons and explicit foreign-return safety."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    import numpy as np
    from research_singleton_free_20261108.full_paths import FULL
    from research_identity_assignment_run_20261108.full_statistics import pooled
    from trackeval.metrics.hota import HOTA
    from evaluation.hota_eval import VALID
    phase('full_phase.json');p=read(FULL/'protocol.json');arms=list(p['parameters']);videos=list(p['videos']);n=len(videos)
    extra=read(FULL/'additional_report.json');assert extra['passed'];points={};samples={};components={};per_video={}
    indices=np.random.default_rng(42).integers(0,n,(20000,n));counts=np.array([np.bincount(v,minlength=n) for v in indices]);family=11
    for arm in arms:
        data=[read(FULL/'metrics'/arm/(v+'.json')) for v in videos];assert all(r['passed'] for r in data)
        components[arm]={k:np.array([r['components'][k] for r in data],float) for k in ['TP','FP','FN','AA']}
        point={k:float(v[0]) for k,v in pooled(components[arm],np.ones((1,n))).items()}
        ids={k:sum(r['components'][k] for r in data) for k in ['IDTP','IDFP','IDFN']}
        point['IDF1']=100*ids['IDTP']/max(1.,ids['IDTP']+.5*ids['IDFP']+.5*ids['IDFN'])
        for key in ['AssA','IDF1']:point['mean_video_'+key]=float(np.mean([r['metrics'][key] for r in data]))
        for key in ['IDSw','CLR_FP','CLR_FN']:point[key]=int(sum(r['metrics'][key] for r in data))
        for key in ['wrong_returns','correct_returns','unknown_returns','reactivations']:point[key]=extra['arms'][arm]['totals'].get(key,0)
        points[arm]=point;samples[arm]=pooled(components[arm],counts)['HOTA'];per_video[arm]={v:r['metrics'] for v,r in zip(videos,data)}
    comparisons={}
    for a,b in p['primary_comparisons']:
        delta=samples[a]-samples[b]
        harm=dict(AssA_mean=points[a]['mean_video_AssA']-points[b]['mean_video_AssA'],IDF1_mean=points[a]['mean_video_IDF1']-points[b]['mean_video_IDF1'],
            IDSw=points[a]['IDSw']-points[b]['IDSw'],FP=points[a]['CLR_FP']-points[b]['CLR_FP'],FN=points[a]['CLR_FN']-points[b]['CLR_FN'],
            wrong_known_returns=points[a]['wrong_returns']-points[b]['wrong_returns'])
        comparisons[a+'-'+b]=dict(delta_pooled_HOTA=points[a]['HOTA']-points[b]['HOTA'],ci95=np.percentile(delta,[2.5,97.5]).tolist(),
            ci_corrected=np.percentile(delta,[2.5/family,100-2.5/family]).tolist(),harm=harm,
            harm_passed=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns']))
    def accepted(a,b):return comparisons[a+'-'+b]['ci_corrected'][0]>0 and comparisons[a+'-'+b]['harm_passed']
    geometry=accepted('U_GEO','B0');threshold=accepted('U_THRESHOLD','B0');signal_system=accepted('U_FLYVIS','B0')
    flyvis=all(accepted('U_FLYVIS',b) for b in arms if b!='U_FLYVIS')
    official=HOTA();checks=0;max_error=0
    for arm in arms:
        detail=[]
        for video in videos:
            with gzip.open(FULL/'metrics'/arm/(video+'.details.json.gz'),'rt',encoding='utf-8') as f:d=json.load(f)[video]
            detail.append({c:{k:np.asarray(v) if isinstance(v,list) else v for k,v in d[c]['HOTA'].items()} for c in VALID if c in d})
        for index,selected in enumerate([np.arange(n),*indices[:20]]):
            combined={}
            for category in VALID:
                seq={str(i):detail[int(j)][category] for i,j in enumerate(selected) if category in detail[int(j)]}
                if seq:combined[category]=official.combine_sequences(seq)
            actual=float(official.combine_classes_det_averaged(combined)['HOTA'].mean()*100)
            expected=points[arm]['HOTA'] if index==0 else float(samples[arm][index-1]);error=abs(actual-expected)
            assert error<1e-9;max_error=max(max_error,error);checks+=1
    with (FULL/'bootstrap_counts.npz').open('xb') as f:np.savez_compressed(f,counts=counts)
    phase('full_phase.json');write(FULL/'comparison_report.json',dict(passed=True,parameters=p['parameters'],videos=videos,pooled=points,
        per_video=per_video,comparisons=comparisons,accepted_geometry=geometry,accepted_threshold=threshold,
        accepted_flyvis_system_and_increment=flyvis,geometry_vs_tuned_threshold_positive=comparisons['U_GEO-U_THRESHOLD']['ci_corrected'][0]>0,
        accepted_system_variants=dict(U_GEO=geometry,U_THRESHOLD=threshold,U_FLYVIS=signal_system),
        flyvis_correction_active=p['parameters']['U_FLYVIS']['beta']!=0,
        bootstrap=dict(samples=20000,seed=42,unit='paired_video',family=family,primary_ci_percent=100*(1-.05/family),official_pooling_checks=checks,max_error=max_error),
        historical=True,independent_final_test=False,
        scope='First problem only. Full causal rollouts; paired-video inference. No additional FlyVis claim from a geometry/threshold gain; problems2–4remain deferred.'),new=True)


if __name__=='__main__':main(guard())
