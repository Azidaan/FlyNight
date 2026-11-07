"""Прежний полный строгий аудит с исправленной операцией NumPy2 и новыми адресами."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    from tracking.history_cost_replay import replay_cost
    folder=RUN/'history_tracker_author_continuation1';p=service.read(folder/'protocol.json');assert service.read(folder/'author_override.json')['accepted']
    source=ROOT/'scripts/audit_history_full_tracker_20261028.py';assert service.sha(source)==p['sha256']['scripts/audit_history_full_tracker_20261028.py']
    code=source.read_text(encoding='utf-8').replace('history_tracker_pilot','history_tracker_author_continuation1').replace('history_tracker_audit','history_tracker_author_audit1')
    limit="end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()"
    assert code.count(limit)==1
    code=code.replace(limit,"end=min(datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp(),datetime.fromisoformat(p['author_audit_absolute_end']).timestamp())")
    old="expected_cost=matrix.dtype.type(old['original']+self.alpha*(.5-1/(1+np.exp(-np.clip(old['score'],-80,80))))) if self.alpha else matrix.dtype.type(old['original'])"
    new="expected_cost=replay_cost(old['original'],old['score'],self.alpha,matrix.dtype)"
    assert code.count(old)==1;code=code.replace(old,new)
    ns={'__name__':'history_author_audit_worker','__file__':str(source),'replay_cost':replay_cost};exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
