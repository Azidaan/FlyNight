"""Official pooled HOTA and fixed twenty-comparison correction, including harms."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    import numpy as np
    from research_identity_assignment_run_20261108.full_statistics import pooled
    from evaluation.hota_eval import VALID
    from trackeval.metrics.hota import HOTA
    full_phase();p=read(FULL/'protocol.json');videos=list(p['videos']);arms=list(p['parameters']);N=len(videos);indices=np.random.default_rng(42).integers(0,N,(20000,N));counts=np.asarray([np.bincount(row,minlength=N) for row in indices]);extra=read(FULL/'additional_report.json');points={};samples={};per_video={};checks=0;max_error=0
    for arm in arms:
        rows=[read(FULL/'metrics'/arm/(v+'.json')) for v in videos];components={k:np.array([r['components'][k] for r in rows],float) for k in ['TP','FP','FN','AA']};point={k:float(value[0]) for k,value in pooled(components,np.ones((1,N))).items()};identities={k:sum(r['components'][k] for r in rows) for k in ['IDTP','IDFP','IDFN']};point['IDF1']=100*identities['IDTP']/max(1,identities['IDTP']+.5*identities['IDFP']+.5*identities['IDFN'])
        for key in ['AssA','IDF1']:point['mean_video_'+key]=float(np.mean([r['metrics'][key] for r in rows]))
        for key in ['IDSw','CLR_FP','CLR_FN']:point[key]=int(sum(r['metrics'][key] for r in rows))
        point.update(extra['arms'][arm]['totals']);points[arm]=point;per_video[arm]={v:r['metrics'] for v,r in zip(videos,rows)};samples[arm]=pooled(components,counts)['HOTA']
        detail=[]
        for video in videos:
            with gzip.open(FULL/'metrics'/arm/(video+'.details.json.gz'),'rt',encoding='utf-8') as f:d=json.load(f)[video]
            detail.append({c:{k:np.asarray(value) if isinstance(value,list) else value for k,value in d[c]['HOTA'].items()} for c in VALID if c in d})
        official=HOTA()
        for number,selection in enumerate([np.arange(N),*indices[:20]]):
            combined={}
            for c in VALID:
                seq={str(i):detail[int(j)][c] for i,j in enumerate(selection) if c in detail[int(j)]}
                if seq:combined[c]=official.combine_sequences(seq)
            value=float(official.combine_classes_det_averaged(combined)['HOTA'].mean()*100);expected=point['HOTA'] if number==0 else samples[arm][number-1];error=abs(value-expected);assert error<1e-9;max_error=max(max_error,error);checks+=1
    comparisons={}
    for a,b in p['primary_comparisons']:
        key=a+'-'+b
        if key in comparisons:continue
        delta=samples[a]-samples[b];harm=dict(AssA_mean=points[a]['mean_video_AssA']-points[b]['mean_video_AssA'],IDF1_mean=points[a]['mean_video_IDF1']-points[b]['mean_video_IDF1'],IDSw=points[a]['IDSw']-points[b]['IDSw'],FP=points[a]['CLR_FP']-points[b]['CLR_FP'],FN=points[a]['CLR_FN']-points[b]['CLR_FN'],wrong_known_returns=points[a].get('wrong_returns',0)-points[b].get('wrong_returns',0));safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns'])
        comparisons[key]=dict(delta_pooled_HOTA=points[a]['HOTA']-points[b]['HOTA'],ci95=np.percentile(delta,[2.5,97.5]).tolist(),ci_corrected=np.percentile(delta,[2.5/20,100-2.5/20]).tolist(),harm=harm,harm_passed=bool(safe))
    accepted={};identity={}
    for arm in p['eligible_systems']:
        c=comparisons[arm+'-B0'];gain=dict(correct_returns=points[arm].get('correct_returns',0)-points['B0'].get('correct_returns',0),sustained_success=points[arm].get('sustained_success',0)-points['B0'].get('sustained_success',0));identity[arm]=gain;accepted[arm]=bool(c['ci_corrected'][0]>0 and c['harm_passed'] and gain['correct_returns']>0 and gain['sustained_success']>0)
    increment=all(comparisons['FLYVIS-'+arm]['ci_corrected'][0]>0 and comparisons['FLYVIS-'+arm]['harm_passed'] for arm in arms if arm!='FLYVIS')
    with (FULL/'bootstrap_counts.npz').open('xb') as f:np.savez_compressed(f,counts=counts)
    full_phase();write(FULL/'comparison_report.json',dict(passed=True,selected_system=p['selected_system'],pooled=points,per_video=per_video,comparisons=comparisons,accepted=accepted,identity_gain=identity,problem1_acceptance=bool(accepted.get(p['selected_system'],False)),secondary_preregistered_systems=accepted,any_preregistered_system_accepted=any(accepted.values()),primary_failed_conditional_screen=p['selected_system'] not in p['eligible_systems'],accepted_flyvis_system_and_increment=bool(accepted.get('FLYVIS',False) and increment),bootstrap=dict(seed=42,repetitions=20000,family=20,unit='paired_video',official_checks=checks,max_error=max_error),historical=True,independent_final_test=False,scope='Causal public-ID linking after actual native confirmation. Exact native physical boxes. Problems2–4not tested. Additional FlyVis contribution separated from whole-system benefit.'),new=True)


if __name__=='__main__':main(guard())
