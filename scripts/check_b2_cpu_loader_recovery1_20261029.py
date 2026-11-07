"""Только CPU-загрузка готовых весов; без обучения, CUDA и обработки видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,time,faulthandler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_cpu_loader_recovery1';folder.mkdir(exist_ok=True)
    import psutil
    proc=psutil.Process();service.write(folder/'launch.json',dict(pid=proc.pid,created=proc.create_time(),updated=service.now()))
    service.write(folder/'status.json',dict(state='CPU checkpoint load',updated=service.now()))
    faulthandler.dump_traceback_later(300,exit=True)
    source=RUN/'b2_sparse_training_pilot';report=service.read(source/'report.json')
    assert report['complete'];path=ROOT/report['final']['path'];assert service.sha(path)==report['final']['sha256']
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    state=torch.load(path,map_location='cpu',weights_only=False)
    from tracking.b2_cpu_loader_recovery1 import load_cpu_branches
    began=time.monotonic();models=load_cpu_branches(state);proof={}
    for name,(branch,bb) in models.items():
        assert all(torch.equal(v,state['models'][name]['branch'][k]) for k,v in branch.state_dict().items()),name
        if name in ['C','D']:actual=bb.net.state_dict()
        elif name=='R1':actual={}
        else:actual=bb.state_dict()
        assert all(torch.equal(v,state['models'][name]['backend'][k]) for k,v in actual.items()),name
        proof[name]=dict(branch_exact=True,backend_exact=True,device='cpu')
    assert service.sha(path)==report['final']['sha256'] and not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(passed=True,training_repeated=False,checkpoint_sha256=service.sha(path),
        arms=proof,no_cuda_initialized=True,seconds=time.monotonic()-began,
        original_failure_sha256=service.sha(RUN/'b2_sparse_training_audit/controller_result.json'),
        source_sha256=service.sha(ROOT/'tracking/b2_cpu_loader_recovery1.py'),updated=service.now()))
    service.write(folder/'status.json',dict(state='complete',updated=service.now()))
    faulthandler.cancel_dump_traceback_later()
    print('CPU checkpoint loading: all six branches exact; no CUDA; no training.')


if __name__=='__main__':main()
