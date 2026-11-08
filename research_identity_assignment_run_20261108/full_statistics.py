"""Pooled official HOTA under paired-video resampling; fixed family and harm guards."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def pooled(component,weights):
    tp=weights@component['TP'];fp=weights@component['FP'];fn=weights@component['FN'];aa=weights@component['AA']
    assa=aa/np.maximum(1,tp);deta=tp/np.maximum(1,tp+fp+fn)
    return dict(HOTA=100*np.sqrt(deta*assa).mean(-1),AssA=100*assa.mean(-1),DetA=100*deta.mean(-1))


def main(obs):
    verify_phase('full_statistics_phase.json');p=read(FULL/'protocol.json');arms=list(p['parameters']);videos=list(p['videos']);n=len(videos)
    all_metrics={};components={};point={};samples={};rng=np.random.default_rng(42)
    indices=rng.integers(0,n,(20000,n));counts=np.stack([np.bincount(row,minlength=n) for row in indices])
    for arm in arms:
        data=[read(FULL/'metrics'/arm/(v+'.json')) for v in videos];assert all(r['passed'] for r in data)
        all_metrics[arm]={v:r['metrics'] for v,r in zip(videos,data)}
        components[arm]={k:np.array([r['components'][k] for r in data],float) for k in ['TP','FP','FN','AA']}
        point[arm]={k:float(v[0]) for k,v in pooled(components[arm],np.ones((1,n))).items()}
        ids={k:sum(r['components'][k] for r in data) for k in ['IDTP','IDFP','IDFN']}
        point[arm]['IDF1']=100*ids['IDTP']/max(1.,ids['IDTP']+.5*ids['IDFP']+.5*ids['IDFN'])
        samples[arm]=pooled(components[arm],counts)['HOTA']
        point[arm]['mean_video_AssA']=float(np.mean([r['metrics']['AssA'] for r in data]));point[arm]['mean_video_IDF1']=float(np.mean([r['metrics']['IDF1'] for r in data]))
        for k in ['IDSw','CLR_FP','CLR_FN']:point[arm][k]=int(sum(r['metrics'][k] for r in data))
    comparisons={}
    for a,b in p['inference']['primary_comparisons']:
        d=samples[a]-samples[b]
        harm=dict(AssA_mean=point[a]['mean_video_AssA']-point[b]['mean_video_AssA'],IDF1_mean=point[a]['mean_video_IDF1']-point[b]['mean_video_IDF1'],
            IDSw=point[a]['IDSw']-point[b]['IDSw'],FP=point[a]['CLR_FP']-point[b]['CLR_FP'],FN=point[a]['CLR_FN']-point[b]['CLR_FN'])
        comparisons[a+'-'+b]=dict(delta_pooled_HOTA=point[a]['HOTA']-point[b]['HOTA'],ci95=np.percentile(d,[2.5,97.5]).tolist(),
            ci99_375=np.percentile(d,[.3125,99.6875]).tolist(),harm=harm,
            harm_passed=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN']))
    geometry=comparisons['G-B0']['ci99_375'][0]>0 and comparisons['G-B0']['harm_passed']
    fly_keys=[key for key in comparisons if key.startswith('G_FLYVIS-')]
    fly=all(comparisons[k]['ci99_375'][0]>0 and comparisons[k]['harm_passed'] for k in fly_keys)
    # Independently validate the vectorized pooling against official combination code.
    from trackeval.metrics.hota import HOTA
    from evaluation.hota_eval import VALID
    official=HOTA();max_error=0.;checks=0
    for arm in arms:
        detail=[]
        for v in videos:
            with gzip.open(FULL/'metrics'/arm/(v+'.details.json.gz'),'rt',encoding='utf-8') as f:d=json.load(f)[v]
            detail.append({c:{k:np.asarray(x) if isinstance(x,list) else x for k,x in d[c]['HOTA'].items()} for c in VALID if c in d})
        for index,selected in enumerate([np.arange(n),*indices[:20]]):
            cls={}
            for c in VALID:
                seq={str(i):detail[int(j)][c] for i,j in enumerate(selected) if c in detail[int(j)]}
                if seq:cls[c]=official.combine_sequences(seq)
            value=float(official.combine_classes_det_averaged(cls)['HOTA'].mean()*100)
            expected=point[arm]['HOTA'] if index==0 else float(samples[arm][index-1])
            error=abs(value-expected);assert error<1e-9;max_error=max(max_error,error);checks+=1
    with (FULL/'bootstrap_counts.npz').open('xb') as f:np.savez_compressed(f,counts=counts)
    report=dict(passed=True,videos=videos,arms=arms,parameters=p['parameters'],pooled=point,per_video=all_metrics,
        comparisons=comparisons,bootstrap=dict(samples=20000,seed=42,unit='paired_video',family=8,primary_ci=99.375,
            endpoint='Official detection-weighted pooled HOTA, recomputed from components for every resample.',official_pooling_checks=checks,max_error=max_error),
        accepted_geometry=geometry,accepted_flyvis_system_and_increment=fly,
        historical=True,independent_final_test=False,
        limits='Same historically studied20val videos. Detector did not train on them, but researchers have seen their results. No new coefficient selection on this pilot.')
    verify_phase('full_statistics_phase.json');write(FULL/'comparison_report.json',report,new=True)


if __name__=='__main__':main(guard())
