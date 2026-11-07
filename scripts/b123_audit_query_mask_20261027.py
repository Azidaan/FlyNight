"""Сверка сохранённых выходов масочного пилота, без повторного расчёта модели."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/b123_20261027/sam2_query_output_pilot'
OUT=ROOT/'runs/b123_20261027/sam2_query_audit'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    end=read(RUN/'controller_result.json');assert end['all_owned_ended']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    if not (RUN/'report.json').exists():
        result=dict(full_pilot_completed=False,passed=False,reason='No complete primary report; preserve partial outputs and ask agent to review',controller=end)
    else:
        assert end['exit_code']==0 and not end['timed_out']
        r=read(RUN/'report.json');p=read(RUN/'protocol.json')
        assert r['complete'] and len(r['videos'])==20 and r['videos']==p['videos']
        assert all(r['baseline_arrays_exact'].values()) and not r['tracker_state_changed']
        assert not r['ltr_reproduced'] and not r['sgar_reproduced'] and not r['flyvis_core_used']
        assert all(x['all_image_requests_causal'] for x in r['counters'].values())
        for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
        controls=read(ROOT/'runs/b123_20261027/recovery_motion_control/report.json')
        busca=read(ROOT/'runs/b123_20261027/busca_output_pilot/report.json')
        hashes={};added=0
        for video in r['videos']:
            a=read(RUN/'baseline'/(video+'.json'))
            original=read(ROOT/'runs/z13_val20_matched_20261024/raw/xywh/raw'/(video+'.json'))
            b=read(RUN/'sam2_mask'/(video+'.json'))
            assert a==original and len(a)==len(b)
            assert r['per_video']['baseline'][video]==controls['per_video']['baseline'][video]
            count=0
            for first,second in zip(a,b):
                assert first['name']==second['name'] and first['videoName']==second['videoName']
                assert second['labels'][:len(first['labels'])]==first['labels']
                assert len({str(x['id']) for x in second['labels']})==len(second['labels'])
                count+=len(second['labels'])-len(first['labels'])
            assert count==r['counters'][video]['added_annotated_boxes'];added+=count
            for arm in ('baseline','sam2_mask'):
                path=RUN/arm/(video+'.json');hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        assert r['pooled']['baseline']==controls['pooled']['baseline']
        contrasts={}
        for name,other in [('vs_baseline',r['per_video']['baseline']),('vs_simple_continuation',controls['per_video']['emit_lost_kalman']),('vs_busca_output',busca['per_video']['busca_output'])]:
            delta=np.array([r['per_video']['sam2_mask'][v]['HOTA']-other[v]['HOTA'] for v in sorted(r['videos'])])
            rng=np.random.default_rng(1)
            values=[delta[index].mean() for index in rng.integers(0,len(delta),(5000,len(delta)))]
            expected=dict(mean_per_video=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),better=int((delta>0).sum()),n=len(delta))
            assert expected==r['paired'][name],name
            rng=np.random.default_rng(27140)
            independent=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)]
            contrasts[name]=dict(reproduced_primary=expected,independent_ci95=np.quantile(independent,[.025,.975]).tolist())
        result=dict(passed=True,full_pilot_completed=True,checked=datetime.now(timezone.utc).isoformat(),
            all20_baseline_json_exact=True,all20_baseline_metrics_exact=True,only_appended_unique_ids=True,
            source_hashes_unchanged=True,owned_processes_ended=True,added_output_labels=added,paired=contrasts,output_sha256=hashes,
            scope='Saved JSON and statistics independently checked; raw array equality asserted by frozen worker, new intermediate arrays not independently saved. Query-only FP32 variant, not dense mask tracking, LTR, SGAR or FlyVis core benefit.')
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    (RUN/'independent_audit.json').write_text(text,encoding='utf-8')
    (OUT/'report.json').write_text(text,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','full_pilot_completed']}))
if __name__=='__main__':main()
