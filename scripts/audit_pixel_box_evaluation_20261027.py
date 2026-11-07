"""Повторить поправки из сохранённых выходов и независимые парные интервалы, без нового инференса видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now

def audit(folder):
    import numpy as np,torch
    from tracking.pixel_box_refinement import make_model,apply_offsets
    run=BASE/folder;p=read(run/'protocol.json');r=read(run/'report.json');controller=read(run/'controller_result.json')
    assert r['complete'] and controller['exit_code']==0 and controller['all_owned_ended'] and not controller['timed_out']
    assert not any(matching(pid,birth) for pid,birth in controller['owned'].items())
    for path,h in p['sha256'].items():assert sha(ROOT/path)==h,path
    torch.set_num_threads(2);is_dev=p['phase']=='dev';learning=read(BASE/'pixel_box_learning/report.json');heads={};specs={};variants={}
    chosen=read(BASE/'pixel_box_frozen_for_validation.json')['models']
    for m in learning['models']:
        assert sha(BASE/m['checkpoint'])==m['sha256'];head=make_model();saved=torch.load(BASE/m['checkpoint'],map_location='cpu',weights_only=True);head.load_state_dict(saved['state_dict']);head.eval();heads[m['name']]=head;specs[m['name']]=m
        for alpha in ([0.,.5,1.] if is_dev else [next(x['alpha'] for x in chosen if x['name']==m['name'])]):variants[m['name']+'_a'+str(alpha)]=dict(name=m['name'],alpha=alpha)
    checked=0;max_diff=0
    for source in p['sources']:
        v=source['video'];base=read(ROOT/source['baseline']);assert read(run/'baseline'/(v+'.json'))==base
        f=run/'offsets'/(v+'.npz');assert sha(f)==r['proof'][v]['offsets_sha256'];offsets=np.load(f);records=read(f.with_name(v+'_records.json'))
        samples=np.load(f.with_name(v+'_audit_samples.npz'));x=torch.from_numpy(samples['images']).float().permute(0,1,4,2,3);x=(x/255-.5)/.25;m=torch.from_numpy(samples['M'])
        for name,head in heads.items():
            delta=offsets[name];assert delta.shape==(len(records),4) and np.isfinite(delta).all() and np.max(abs(delta),initial=0)<=.250001
            if len(x):
                with torch.inference_mode():repeat=head(x,m,specs[name]['arm']=='temporal').numpy()
                difference=float(np.max(abs(repeat-delta[samples['indices']])));max_diff=max(max_diff,difference);assert difference<2e-5,(v,name,difference)
        assert all(a['past_frame'] is None or a['past_frame']<a['frame'] for a in records)
        for name,var in variants.items():
            expected=copy.deepcopy(base)
            if var['alpha']:
                lookup=[{str(x['id']):x for x in row['labels']} for row in expected]
                for rec,delta in zip(records,offsets[var['name']]):
                    if not np.any(delta):continue
                    b=apply_offsets(rec['box'],delta,var['alpha']);lookup[rec['reference']][str(rec['id'])]['box2d']={k:float(value) for k,value in zip(['x1','y1','x2','y2'],b)}
            actual=read(run/name/(v+'.json'));assert actual==expected,(v,name)
            if var['alpha']==0:assert actual==base and r['per_video'][name][v]==r['per_video']['baseline'][v]
            checked+=len(actual)
        if not is_dev:assert r['per_video']['baseline'][v]==source['baseline_metrics']
    independent={};decision={}
    if is_dev:
        for model in chosen:
            names=[model['name']+'_a'+str(a) for a in [0.,.5,1.]]
            scores={name:float(np.mean([x['HOTA'] for x in r['per_video'][name].values()])) for name in names}
            best=max([0.,.5,1.],key=lambda alpha:(scores[model['name']+'_a'+str(alpha)],-alpha));assert model['alpha']==best
        assert r['chosen']==chosen and not r['val_used']
    else:
        assert r['fixed_models']==chosen and not r['weights_or_alpha_changed_on_val'] and not r['ground_truth_used_by_predictor']
        rng=np.random.default_rng(281028)
        for group,pairs in r['paired'].items():
            vs=[s['video'] for s in p['sources']] if group=='all40' else p['groups'][group];independent[group]={}
            for key,value in pairs.items():
                if key.startswith('temporal_vs_current_seed'):
                    seed=key.removeprefix('temporal_vs_current_seed');a=next(n for n in variants if variants[n]['name']=='temporal_seed'+seed);b=next(n for n in variants if variants[n]['name']=='current_seed'+seed)
                else:a=key.removesuffix('_vs_baseline');b='baseline'
                d=np.array([r['per_video'][a][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in vs]);assert abs(d.mean()-value['mean_per_video'])<1e-10
                assert value['n']==len(vs) and value['better']==int((d>0).sum())
                boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(1);independent[group][key]=dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(vs))
            for arm in ['current','temporal']:
                decision[group+'_'+arm+'_all3_vs_baseline']=all(pairs[next(n for n in variants if variants[n]['name']==arm+'_seed'+str(seed))+'_vs_baseline']['ci95'][0]>0 for seed in [42,43,44])
            decision[group+'_temporal_all3_vs_current']=all(pairs['temporal_vs_current_seed'+str(seed)]['ci95'][0]>0 for seed in [42,43,44])
    result=dict(passed=True,phase=p['phase'],videos=len(p['sources']),checked_output_frames=checked,only_boxes_changed=True,null_exact=True,strict_past=True,
        checkpoint_hashes_verified=True,source_hashes_verified=True,max_cpu_gpu_difference=max_diff,independent_pairs=independent,prespecified_decision=decision,
        report_sha256=sha(run/'report.json'),updated=now())
    write(run/'independent_audit.json',result);write(BASE/(folder+'_audit')/'report.json',result)
    print('Geometry evaluation independently audited: '+str(decision))
    return result
