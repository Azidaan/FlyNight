"""Независимая CPU-сверка нового правила выдачи, без повторной оценки модели."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/b123_20261027';RUN=BASE/'detection_supported_recovery';OUT=BASE/'detection_supported_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    end=read(RUN/'controller_result.json');assert end['all_owned_ended']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    if not (RUN/'report.json').exists():result=dict(passed=False,complete=False,reason='No full primary report; preserve partial result',controller=end)
    else:
        assert end['exit_code']==0 and not end['timed_out'];r=read(RUN/'report.json');p=read(RUN/'protocol.json')
        assert r['complete'] and len(r['videos'])==20 and r['videos']==p['videos'] and not r['tracker_state_changed'] and not r['new_inference']
        for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
        baseline=read(BASE/'recovery_motion_control/report.json');hashes={};counts={arm:0 for arm in p['arms'] if arm!='baseline'}
        for v in r['videos']:
            original=read(ROOT/f'runs/z13_val20_matched_20261024/raw/xywh/raw/{v}.json');base=read(RUN/'baseline'/(v+'.json'));assert original==base
            assert r['per_video']['baseline'][v]==baseline['per_video']['baseline'][v]
            for arm in counts:
                frames=read(RUN/arm/(v+'.json'));assert len(frames)==len(base)
                added=0
                for a,b in zip(base,frames):
                    assert a['name']==b['name'] and a['videoName']==b['videoName'] and b['labels'][:len(a['labels'])]==a['labels']
                    assert len({str(x['id']) for x in b['labels']})==len(b['labels']);added+=len(b['labels'])-len(a['labels'])
                assert added==r['counters'][v][arm]['accepted'];counts[arm]+=added
                path=RUN/arm/(v+'.json');hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        assert r['pooled']['baseline']==baseline['pooled']['baseline']
        contrasts={}
        for name,a,b in [('mask_vs_baseline','mask_support','baseline'),('kalman_vs_baseline','kalman_support','baseline'),('mask_vs_kalman','mask_support','kalman_support')]:
            delta=np.array([r['per_video'][a][v]['HOTA']-r['per_video'][b][v]['HOTA'] for v in sorted(r['videos'])])
            primary=delta[np.random.default_rng(1).integers(0,20,(5000,20))].mean(1)
            expected=dict(mean_per_video=float(delta.mean()),ci95=np.quantile(primary,[.025,.975]).tolist(),better=int((delta>0).sum()),n=20)
            assert expected==r['paired'][name]
            samples=[np.random.default_rng(seed).choice(delta,20,replace=True).mean() for seed in range(27160,32160)]
            contrasts[name]=dict(reproduced_primary=expected,independent_ci95=np.quantile(samples,[.025,.975]).tolist())
        result=dict(passed=True,complete=True,checked=datetime.now(timezone.utc).isoformat(),all20_baseline_json_exact=True,all20_baseline_metrics_exact=True,
            only_appended_unique_ids=True,frozen_inputs_unchanged=True,controller_ended=True,added_labels=counts,paired=contrasts,output_sha256=hashes)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n';(RUN/'independent_audit.json').write_text(text,encoding='utf-8');(OUT/'report.json').write_text(text,encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],complete=result['complete'])))
if __name__=='__main__':main()
