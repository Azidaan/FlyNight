"""Only if the immutable internal gate passes: separately bounded final18heads on26videos."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import z13_temporal_identity_vectorized_recovery2_20261025 as V
from scripts.z13_temporal_identity_vectorized_recovery3_20261025 import structural_stack
S=V.S;OUT=S.RUN/'final_execution'


def prepare():
    assert not OUT.exists()
    r=S.read(S.RUN/'report.json');c=S.read(S.RUN/'recovery2/controller_result.json');p=S.read(S.RUN/'protocol.json')
    assert r['internal_pass'] and r['final_pending'] and c['exit_code']==0 and c['all_owned_ended']
    start=S.now();soft=min(start+timedelta(seconds=p['final_budget_seconds']),datetime.fromisoformat(p['save_at']))
    V.retry_write(OUT/'protocol.json',dict(device='cuda',budget_seconds=p['final_budget_seconds'],save_at=soft.isoformat(),hard_stop=(soft+timedelta(seconds=20)).isoformat(),stop_request_path=(OUT/'stop.request').relative_to(ROOT).as_posix(),source_sha256=S.sha(Path(__file__)),internal_report_sha256=S.sha(S.RUN/'internal_report.json'),authorization='Only preregistered final18models after all six internal contrasts pass; no HOTA.'))
    V.retry_write(OUT/'status.json',dict(state='ready',updated=start.isoformat()))


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.set_default_device('cpu');torch.func.stack_module_state=lambda models:structural_stack(models,torch)
    p=S.read(S.RUN/'protocol.json');q=S.read(OUT/'protocol.json');assert S.sha(Path(__file__))==q['source_sha256']
    r=S.read(S.RUN/'report.json');assert r['internal_pass'] and S.sha(S.RUN/'internal_report.json')==q['internal_report_sha256']
    deadline=datetime.fromisoformat(q['save_at']);V.OUT=OUT;results={};hashes=dict(r['models_report_sha256']);began=time.monotonic()
    split=dict(train=p['fit84'],calibration=p['fit20'],validation=p['evaluate26'])
    with GpuLock('flyvis_fovea_gpu'):
        for number,arm in enumerate(S.ARMS):
            cpu,videos=S.load_data(torch,arm,include26=True);data={k:v.to('cuda:0') for k,v in cpu.items()}
            descriptors=[dict(seed=s,split=split,folder=str(S.RUN/'final'/arm/f'seed{s}')) for s in p['seeds']]
            records,elapsed=V.train_group(torch,p,cpu,data,videos,descriptors,arm,deadline,0,'final')
            for d,rec in zip(descriptors,records):results[arm,d['seed']]=rec['per_video'];folder=Path(d['folder']);hashes[folder.relative_to(ROOT).as_posix()]=S.sha(folder/'report.json')
            if number==0:
                estimate=2*elapsed/3*15+60;left=(deadline-S.now()).total_seconds();V.retry_write(OUT/'price.json',dict(estimate=estimate,remaining=left,proceed=estimate<left));assert estimate<left,'Original final600second budget exceeded'
            del cpu,data;gc.collect();torch.cuda.empty_cache()
    means={a:{v:float(np.mean([results[a,s][v] for s in p['seeds']])) for v in p['evaluate26']} for a in S.ARMS}
    comparisons=S.contrast(means);comparisons['D-original_context']=V.old_contrast(means['D'],S.read(S.FOLLOW/'training/report.json')['mean_seed_per_video']['D_context'])
    passed=S.passes(comparisons)
    V.retry_write(S.RUN/'final_report.json',dict(complete=True,means=means,paired=comparisons,primary_pass=passed,new26evaluations=1))
    r.update(models=108,final_pass=passed,final_evaluated=True,final_pending=False,models_report_sha256=hashes,final_seconds=time.monotonic()-began)
    V.retry_write(S.RUN/'report.json',r);V.retry_write(S.RUN/'status.json',dict(state='completed',models_done=108,updated=S.now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:V.retry_write(OUT/'failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise
