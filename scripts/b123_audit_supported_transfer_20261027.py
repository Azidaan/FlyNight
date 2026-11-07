"""Сверка парного измерения на другой части прежнего val."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/b123_20261027';RUN=BASE/'supported_transfer';OUT=BASE/'supported_transfer_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main():
    end=read(RUN/'controller_result.json');assert end['all_owned_ended']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    if not (RUN/'report.json').exists():result=dict(passed=False,complete=False,reason='No complete report; save partial run and inspect cost/error',controller=end)
    else:
        assert end['exit_code']==0 and not end['timed_out'];r=read(RUN/'report.json');p=read(RUN/'protocol.json')
        assert r['complete'] and len(r['videos'])==20 and r['videos']==p['videos'] and not set(r['videos'])&set(p['development_videos'])
        assert not r['flyvis_core_used'] and not r['sam2_used'] and not r['tracker_state_changed']
        for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
        hashes={};added=0
        for v in r['videos']:
            folder=RUN/'cache'/v;proof=r['proof'][v];assert proof['disabled_exact_every_frame'] and proof['one_shared_detection_cache']
            assert sha(folder/'raw_video.npz')==proof['detector_cache_sha256'] and sha(folder/'tracker_arrays.npz')==proof['tracker_arrays_sha256']
            for item in p['images'][v]:assert sha(ROOT/item['path'])==item['sha256']
            a=read(RUN/'baseline'/(v+'.json'));b=read(RUN/'supported'/(v+'.json'));assert len(a)==len(b)==len(p['images'][v])
            alignment=read(folder/'alignment.json')['matches'];assert len(alignment)==len(a)
            with np.load(folder/'tracker_arrays.npz') as z:
                baseline=z['baseline'];supported=z['supported_queries'];assert np.isfinite(baseline).all() and np.isfinite(supported).all()
                for first,second,match in zip(a,b,alignment):
                    frame=int(match['video_frame_index']);raw=baseline[baseline[:,0]==frame,1:];extra=supported[supported[:,0]==frame,1:]
                    assert np.array_equal(raw,extra[:len(raw),:7])
                    assert first['name']==second['name'] and second['labels'][:len(first['labels'])]==first['labels']
                    assert len({str(x['id']) for x in second['labels']})==len(second['labels']);added+=len(second['labels'])-len(first['labels'])
            for arm in ['baseline','supported']:hashes[(RUN/arm/(v+'.json')).relative_to(ROOT).as_posix()]=sha(RUN/arm/(v+'.json'))
        delta=np.array([r['per_video']['supported'][v]['HOTA']-r['per_video']['baseline'][v]['HOTA'] for v in sorted(r['videos'])])
        primary=delta[np.random.default_rng(1).integers(0,20,(5000,20))].mean(1)
        expected=dict(mean_per_video=float(delta.mean()),ci95=np.quantile(primary,[.025,.975]).tolist(),better=int((delta>0).sum()),n=20);assert expected==r['paired']
        rng=np.random.default_rng(27200);independent=[rng.choice(delta,20,replace=True).mean() for _ in range(5000)]
        result=dict(passed=True,complete=True,checked=datetime.now(timezone.utc).isoformat(),disjoint_from_rule_development=True,
            never_used_project_test=False,frozen_inputs_unchanged=True,disabled_control_raw_arrays_preserved=True,only_appended_unique_ids=True,
            added_output_labels=added,paired_reproduced=expected,independent_ci95=np.quantile(independent,[.025,.975]).tolist(),output_sha256=hashes)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n';(RUN/'independent_audit.json').write_text(text,encoding='utf-8');(OUT/'report.json').write_text(text,encoding='utf-8');print(json.dumps(dict(passed=result['passed'],complete=result['complete'])))
if __name__=='__main__':main()
