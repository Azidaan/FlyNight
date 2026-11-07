"""Сверить фиксированный вывод, решения из вероятностей, CPUповтор головы и парные интервалы."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching
RUN=BASE/'learned_validation'

def audit():
    p=read(RUN/'protocol.json');r=read(RUN/'report.json');controller=read(RUN/'controller_result.json')
    assert r['complete'] and controller['exit_code']==0 and controller['all_owned_ended'] and not controller['timed_out']
    assert not any(matching(k,v) for k,v in controller['owned'].items())
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    assert r['fixed_models']==read(BASE/'frozen_for_validation.json')['models']==p['models']
    assert not r['weights_changed_on_val'] and not r['ground_truth_used_by_predictor'] and r['batchnorm_buffers_unchanged']
    import torch
    torch.set_num_threads(2)
    from tracking.learned_recovery_admission import AdmissionModel
    specs={Path(m['checkpoint']).stem:m for m in p['models']}
    heads={name:AdmissionModel(BASE/'admission_learning'/m['checkpoint'],BASE/'admission_learning'/m['normalization'],'cpu') for name,m in specs.items()}
    checked_rows=0;max_cpu_gpu_difference=0.
    for source in p['sources']:
        video=source['video'];expected=read(ROOT/source['baseline']);expected_rule=read(ROOT/source['rule'])
        base=read(RUN/'baseline'/(video+'.json'));rule=read(RUN/'rule'/(video+'.json'))
        assert base==expected and rule==expected_rule
        assert r['per_video']['baseline'][video]==source['baseline_metrics'] and r['per_video']['rule'][video]==source['rule_metrics']
        f=RUN/'features'/(video+'.npz');assert sha(f)==r['proof'][video]['features_sha256'];features=np.load(f)
        records=read(RUN/'features'/(video+'_records.json'));assert len(features['X'])==len(features['E'])==len(records)
        assert np.isfinite(features['X']).all() and np.isfinite(features['E']).all() and all(x['past_frame']<x['frame'] for x in records)
        for name,spec in specs.items():
            probability=np.load(RUN/'features'/(video+'_'+name+'_probabilities.npy'));assert len(probability)==len(records) and np.isfinite(probability).all()
            if len(records):
                indices=np.unique(np.array([0,len(records)//2,len(records)-1]))
                repeated=heads[name].probabilities(features['X'][indices],features['E'][indices])
                error=float(np.max(abs(repeated-probability[indices])));max_cpu_gpu_difference=max(max_cpu_gpu_difference,error);assert error<2e-5
            allow={}
            for record,score in zip(records,probability):
                if score>=spec['threshold']:allow.setdefault(record['reference'],set()).add(record['id'])
            actual=read(RUN/name/(video+'.json'));assert len(actual)==len(base)
            for ref,(a,b,s) in enumerate(zip(actual,base,rule)):
                assert {k:v for k,v in a.items() if k!='labels'}=={k:v for k,v in b.items() if k!='labels'}
                extras=[c for c in s['labels'][len(b['labels']):] if c['id'] in allow.get(ref,set())]
                assert a['labels']==b['labels']+extras and len({c['id'] for c in a['labels']})==len(a['labels']);checked_rows+=1
    rng=np.random.default_rng(281027);independent={};baseline_means={};decision={}
    for group,reported in r['paired'].items():
        videos=list(r['per_video']['baseline']) if group=='all40' else p['groups'][group]
        baseline_means[group]={control:float(np.mean([r['per_video'][control][v]['HOTA'] for v in videos])) for control in ['baseline','rule']}
        independent[group]={}
        for key,values in reported.items():
            if key.startswith('appearance_vs_numeric_seed'):
                seed=key.removeprefix('appearance_vs_numeric_seed');a='appearance_seed'+seed;b='numeric_seed'+seed
            else:a,b=key.rsplit('_vs_',1)
            d=np.array([r['per_video'][a][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in videos])
            assert abs(float(d.mean())-values['mean_per_video'])<1e-10 and values['n']==len(videos) and values['better']==int((d>0).sum())
            boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
            independent[group][key]=dict(mean_per_video=float(d.mean()),ci95=ci,n=len(videos))
        for arm in ['numeric','appearance']:
            decision[group+'_'+arm+'_all3_vs_both']=all(reported[arm+'_seed'+str(seed)+'_vs_'+control]['ci95'][0]>0 for seed in [42,43,44] for control in ['baseline','rule'])
        decision[group+'_appearance_all3_vs_numeric']=all(reported['appearance_vs_numeric_seed'+str(seed)]['ci95'][0]>0 for seed in [42,43,44])
    result=dict(passed=True,complete_videos=40,checked_output_frames=checked_rows,source_hashes_verified=True,
        fixed_weights_and_thresholds_exact=True,base_and_rule_json_exact=True,all_decisions_reconstructed=True,strict_past_all_records=True,
        cpu_gpu_head_max_difference=max_cpu_gpu_difference,independent_seed=281027,independent_pairs=independent,
        baseline_mean_per_video=baseline_means,prespecified_decision=decision,controller_all_owned_ended=True,
        report_sha256=sha(RUN/'report.json'),source_sha256=sha(__file__),updated=datetime.now(timezone.utc).isoformat())
    write(RUN/'independent_audit.json',result)
    print(json.dumps(dict(passed=True,decision=decision,max_cpu_gpu_difference=max_cpu_gpu_difference),ensure_ascii=False))
    return result

if __name__=='__main__':audit()
