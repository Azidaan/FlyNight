"""Прежний строгий аудит, явные CPU-устройства, общий срок без меньшего лимита."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    from scripts.history_stage_budget_policy_20261029 import policy
    policy()
    old=RUN/'b2_sparse_training_audit';c=service.read(old/'controller_result.json')
    assert c['exit_code']==1 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    folder=RUN/'b2_sparse_training_audit_recovery2';p=service.read(folder/'protocol.json')
    assert service.sha(old/'controller_result.json')==p['original_failure_sha256']
    smoke=RUN/'b2_cpu_loader_recovery2/report.json'
    assert service.sha(smoke)==p['cpu_loading_proof_sha256'] and service.read(smoke)['passed']
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    assert service.sha(source)==p['sha256'][source.relative_to(ROOT).as_posix()]
    from tracking.b2_cpu_loader_recovery2 import load_cpu_branches
    code=source.read_text(encoding='utf-8').replace('b2_sparse_training_audit','b2_sparse_training_audit_recovery2')
    assert code.count("models=load_branches(state,'cpu')")==1
    code=code.replace("models=load_branches(state,'cpu')","models=load_cpu_branches(state)")
    assert code.count('if need>=available:')==1
    code=code.replace('if need>=available:','if False:  # Author cancelled stage time gates; forecast retained')
    ns={'__name__':'b2_strict_audit_cpu_recovery2','__file__':str(source),'load_cpu_branches':load_cpu_branches}
    exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
