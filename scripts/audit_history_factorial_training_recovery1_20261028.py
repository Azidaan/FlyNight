"""Прежние строгие проверки с отдельными адресами восстановленной серии."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    original=ROOT/'scripts/audit_history_factorial_training_20261028.py'
    p=service.read(RUN/'history_factorial_training_recovery1/protocol.json')
    assert service.sha(original)==p['sha256']['scripts/audit_history_factorial_training_20261028.py']
    code=original.read_text(encoding='utf-8')
    code=code.replace('history_factorial_training_audit','history_factorial_training_audit_recovery1')
    code=code.replace('history_factorial_training/','history_factorial_training_recovery1/')
    code=code.replace("RUN/'history_factorial_training'","RUN/'history_factorial_training_recovery1'")
    code=code.replace('from tracking.history_pair_model import HistoryPairHead','from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead as HistoryPairHead')
    ns={'__name__':'history_factorial_recovery_audit','__file__':str(original)}
    exec(compile(code,str(original),'exec'),ns);ns['main']()


if __name__=='__main__':main()
