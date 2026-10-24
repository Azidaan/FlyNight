"""Finish and verify recovered controls without polling the language model."""
from pathlib import Path
from datetime import datetime
import os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_weekend_driver_20261024 import ready,execute,verify_and_graph
W=ROOT/'runs/z13_weekend_20261024';REC=ROOT/'runs/z13_matched_trackers_20261024/stock_recovery1'


def main():
    import msvcrt
    lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),source_sha256=sha(Path(__file__))))
    deadline=datetime.fromisoformat(read(REC/'protocol.json')['hard_stop'])
    while not (REC/'controller_result.json').exists():
        if now()>deadline:raise TimeoutError('Recovery result absent after original hard deadline')
        time.sleep(5)
    while not ready(REC):time.sleep(1)
    if not (REC/'finish_recorded.json').exists():execute('record_stock_recovery1',['scripts/z13_record_stock_recovery1_20261024.py'])
    verify_and_graph('stock_recovery1');write(W/'stock_recovery1.json',dict(at=now().isoformat(),complete=True,previous_stock_failure_preserved=True))


if __name__=='__main__':
    try:main()
    except Exception as e:write(W/'stock_recovery1_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
