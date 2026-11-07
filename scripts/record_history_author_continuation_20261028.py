from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    p=service.read(RUN/'history_tracker_author_continuation1/protocol.json')
    source=ROOT/'scripts/record_history_full_tracker_20261028.py';assert service.sha(source)==p['sha256']['scripts/record_history_full_tracker_20261028.py']
    code=source.read_text(encoding='utf-8').replace('history_tracker_pilot','history_tracker_author_continuation1').replace('history_tracker_audit','history_tracker_author_audit1').replace('history_tracker_record','history_tracker_author_record1')
    ns={'__name__':'history_author_record_worker','__file__':str(source)};exec(compile(code,str(source),'exec'),ns);ns['main']()


if __name__=='__main__':main()
