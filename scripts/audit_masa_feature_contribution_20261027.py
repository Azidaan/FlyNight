"""Независимо проверить отключённый/перемешанный признак и полный набор выходов."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,matching,now

def main():
    import numpy as np,joblib
    from scripts.run_masa_feature_contribution_20261027 import merge_map,apply_map
    run=BASE/'masa_feature_contribution';p=read(run/'protocol.json');r=read(run/'report.json');c=read(run/'controller_result.json')
    assert r['complete'] and c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    model=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib')['model'];count=0
    for v in p['videos']:
        z=np.load(run/(v+'_features.npz'));x=z['X'];sig=np.load(ROOT/f'runs/z13_val20_matched_20261024/xywh/signatures/{v}/signatures.npz');raw=read(run/'raw'/(v+'.json'))
        assert np.array_equal(model.predict_proba(x)[:,1] if len(x) else np.zeros(0),z['full_scores'])
        full,links=merge_map(sig,z['pairs'],z['full_scores']);assert apply_map(raw,full)==read(run/'full'/(v+'.json'));assert len(links)==r['proof'][v]['full_links']
        for arm in ['neutral_masa','shuffled42','shuffled43','shuffled44']:
            xx=x.copy()
            if arm=='neutral_masa':xx[:,-1]=model[0].mean_[-1]
            else:xx[:,-1]=x[np.random.default_rng(int(arm.removeprefix('shuffled'))).permutation(len(x)),-1]
            assert np.array_equal(xx[:,:-1],x[:,:-1]);score=model.predict_proba(xx)[:,1] if len(xx) else np.zeros(0);roots,accepted=merge_map(sig,z['pairs'],score)
            assert apply_map(raw,roots)==read(run/arm/(v+'.json'));assert len(accepted)==r['proof'][v]['arms'][arm]['links'];assert sha(run/arm/(v+'.json'))==r['proof'][v]['arms'][arm]['output_sha256'];count+=1
        write(BASE/'masa_feature_contribution_audit/status.json',dict(state='running',updated=now(),video=v,checked_outputs=count),required=False)
    independent={};rng=np.random.default_rng(271028)
    for name,value in r['paired'].items():
        a,b=name.split('_vs_');d=np.array([r['per_video'][a][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in p['videos']]);assert abs(d.mean()-value['mean_per_video'])<1e-10 and value['n']==20
        boot=d[rng.integers(0,20,(5000,20))].mean(1);independent[name]=dict(mean_per_video=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=20)
    result=dict(passed=True,complete_videos=20,checked_control_outputs=count,all_control_outputs_replayed=True,unchanged_other_features=True,source_hashes_verified=True,
        independent_pairs=independent,feature_added_value_supported=r['feature_added_value_supported'],report_sha256=sha(run/'report.json'),updated=now())
    write(run/'independent_audit.json',result);write(BASE/'masa_feature_contribution_audit/report.json',result);write(BASE/'masa_feature_contribution_audit/status.json',dict(state='completed',updated=now()));print('MASA feature intervention audited')

if __name__=='__main__':main()
