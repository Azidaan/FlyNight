"""Те же строгие проверки; отдельно зарегистрированный предел2эпохи."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    source=ROOT/'scripts/audit_history_factorial_training_20261028.py';folder=RUN/'history_two_epoch_pilot'
    p=service.read(folder/'protocol.json');assert service.sha(source)==p['sha256']['scripts/audit_history_factorial_training_20261028.py']
    code=source.read_text(encoding='utf-8')
    code=code.replace('history_factorial_training_audit','history_two_epoch_audit').replace('history_factorial_training/','history_two_epoch_pilot/').replace("RUN/'history_factorial_training'","RUN/'history_two_epoch_pilot'")
    code=code.replace('history_factorial_training_v1','history_factorial_two_epoch_pilot_v1').replace('/6эпох','/2эпохи')
    code=code.replace("cannot_claim='Это неHOTA", "cannot_claim='Только2эпохи,не6;возможное недообучение не исключено. Это неHOTA")
    code=code.replace('from tracking.history_pair_model import HistoryPairHead','from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead as HistoryPairHead')
    old="checkpoint['protocol_sha256']==sha(source/'protocol.json')"
    new="(checkpoint['protocol_sha256']==sha(source/'protocol.json') or (m['variant']=='0' and m['seed']==42 and checkpoint['epoch']==1 and sha(out/'best.pt')==p['reused_epoch']['best_sha256'] and checkpoint['protocol_sha256']==p['reused_epoch']['protocol_sha256']))"
    assert code.count(old)==1;code=code.replace(old,new)
    ns={'__name__':'history_two_epoch_audit','__file__':str(source)};exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
