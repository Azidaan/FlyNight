"""Техническое восстановление в остатке прежнего075окна; отдельные источники."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime
import time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    profile=RUN/'history_deterministic_pool_profile';c=service.read(profile/'controller_result.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert service.read(profile/'report.json')['passed']
    original=ROOT/'scripts/train_history_factorial_20261028.py'
    p=service.read(RUN/'history_factorial_training_recovery1/protocol.json')
    assert service.sha(original)==p['sha256']['scripts/train_history_factorial_20261028.py']
    end=datetime.fromisoformat(p['original_absolute_deadline']).timestamp()
    assert datetime.fromisoformat(service.read(RUN/'history_factorial_training_recovery1/execution_deadline.json')['deadline']).timestamp()<=end-900
    code=original.read_text(encoding='utf-8')
    old='from tracking.history_pair_model import HistoryPairHead,weighted_pair_loss'
    assert code.count(old)==1
    code=code.replace(old,'from tracking.history_pair_model import weighted_pair_loss\n    from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead as HistoryPairHead')
    assert code.count("folder=RUN/'history_factorial_training'")==1
    code=code.replace("folder=RUN/'history_factorial_training'","folder=RUN/'history_factorial_training_recovery1'")
    ns={'__name__':'history_factorial_recovery_worker','__file__':str(original)}
    exec(compile(code,str(original),'exec'),ns);ns['main']()
    # Очередь запустит аудит только после фактического конца этогоGPUдерева.
    budget=min(900,int(end-time.time()-45));assert budget>120
    folder=RUN/'history_factorial_training_audit_recovery1';script='scripts/audit_history_factorial_training_recovery1_20261028.py'
    service.write(folder/'protocol.json',dict(budget_seconds=budget,parent_experiment='history_factorial_training_v1',original_absolute_deadline=p['original_absolute_deadline'],charged_to='Reserved part of original075absolute window; old080 remains failed and is not resumed',sha256={script:service.sha(ROOT/script)}))
    service.write(RUN/'ready_jobs/078_history_factorial_training_audit_recovery1.json',dict(name='078_history_factorial_training_audit_recovery1',folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))


if __name__=='__main__':main()
