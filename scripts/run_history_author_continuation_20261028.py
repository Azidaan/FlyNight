"""Продолжение возможно только после отдельного явного решения автора."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'history_tracker_author_continuation1';a=service.read(folder/'author_override.json')
    assert a['accepted'] is True and a['human_message'] and a['scope']=='Continue same frozen32tracker matrix beyond original090cost gate with new explicit stage deadlines'
    p=service.read(folder/'protocol.json');assert p['author_override_sha256']==service.sha(folder/'author_override.json')
    source=ROOT/'scripts/run_history_full_tracker_pilot_20261028.py';assert service.sha(source)==p['sha256']['scripts/run_history_full_tracker_pilot_20261028.py']
    code=source.read_text(encoding='utf-8').replace('history_tracker_pilot','history_tracker_author_continuation1')
    old="end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()"
    assert code.count(old)==1
    code=code.replace(old,"end=min(datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp(),datetime.fromisoformat(p['author_gpu_absolute_end']).timestamp())")
    ns={'__name__':'history_author_gpu_worker','__file__':str(source)};exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
