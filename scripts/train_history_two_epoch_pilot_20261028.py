"""Отдельный двухэпоховый пилот; не выдаёт ценовой стоп6эпох за успех."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime
import time,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    source=ROOT/'scripts/train_history_factorial_20261028.py';folder=RUN/'history_two_epoch_pilot'
    p=service.read(folder/'protocol.json');assert service.sha(source)==p['sha256']['scripts/train_history_factorial_20261028.py']
    price=service.read(folder/'price.json');assert price['proceed'] and price['first_epoch_reused']
    code=source.read_text(encoding='utf-8').replace("folder=RUN/'history_factorial_training'","folder=RUN/'history_two_epoch_pilot'")
    code=code.replace('from tracking.history_pair_model import HistoryPairHead,weighted_pair_loss','from tracking.history_pair_model import weighted_pair_loss\n    from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead as HistoryPairHead')
    ns={'__name__':'history_two_epoch_worker','__file__':str(source)};exec(compile(code,str(source),'exec'),ns);ns['main']()
    end=datetime.fromisoformat(p['original_absolute_deadline']).timestamp();budget=min(900,int(end-time.time()-45));assert budget>120
    f=RUN/'history_two_epoch_audit';s='scripts/audit_history_two_epoch_pilot_20261028.py'
    service.write(f/'protocol.json',dict(budget_seconds=budget,original_absolute_deadline=p['original_absolute_deadline'],parent_experiment='history_factorial_two_epoch_pilot_v1',sha256={s:service.sha(ROOT/s)}))
    service.write(RUN/'ready_jobs/082_history_two_epoch_audit.json',dict(name='082_history_two_epoch_audit',folder=f.relative_to(ROOT).as_posix(),script=s,script_sha256=service.sha(ROOT/s),protocol_sha256=service.sha(f/'protocol.json')))


if __name__=='__main__':main()
