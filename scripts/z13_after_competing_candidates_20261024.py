"""Complete current CPU pilot and verification without repeated model calls."""
from pathlib import Path
from datetime import datetime
import os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_weekend_driver_20261024 import ready,execute,verify_and_graph
RUN=ROOT/'runs/z13_competing_candidates_20261024';W=ROOT/'runs/z13_weekend_20261024'


def main():
    import msvcrt
    lock=(RUN/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();write(RUN/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),source_sha256=sha(Path(__file__))))
    while not (RUN/'controller_result.json').exists():
        if now()>=datetime.fromisoformat(read(RUN/'controller.json')['hard_stop']):raise TimeoutError('No result before CPU hard deadline')
        time.sleep(5)
    while not ready(RUN):time.sleep(1)
    if not (RUN/'finish_recorded.json').exists():execute('record_competing_candidates',['scripts/z13_record_competing_candidates_20261024.py'])
    verify_and_graph('competing_candidates');write(W/'competing_candidates.json',dict(at=now().isoformat(),complete=True))


if __name__=='__main__':
    try:main()
    except Exception as e:write(W/'competing_candidates_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
