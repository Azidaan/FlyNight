"""Dispatch the read-only full audit after the verified full executor exits."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    full=RUN/'full';began=time.monotonic()
    while not (RUN/'full_audit_wait_deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'full_audit_wait_deadline_registered.json').exists()
    next_receipt=read(RUN/'next_stage_started.json');next_process=psutil.Process(next_receipt['pid'])
    assert abs(next_process.create_time()-next_receipt['created'])<.01
    while not (full/'manager_started.json').exists():
        if (RUN/'next_stage_result.json').exists() and not read(RUN/'next_stage_result.json')['full_pilot_dispatched']:return
        if not next_process.is_running() and not (full/'manager_started.json').exists():
            write(RUN/'full_audit_wait_stopped.json',dict(reason='Earlier executor ended before full registration; recover its failure first.'),new=True);return
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    receipt=read(full/'manager_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (full/'analyses_complete.json').exists():
        if not process.is_running():
            write(RUN/'full_audit_wait_stopped.json',dict(reason='Full manager ended before complete analyses; preserve and recover its failure.'),new=True);return
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    phase('full_phase.json')
    from research_singleton_free_20261108.full_manager import launch
    launch('independent_audit','full_audit.py',[])
    assert read(full/'independent_audit.json')['passed']
    write(RUN/'full_audit_queue_result.json',dict(passed=True,audit_sha256=sha(full/'independent_audit.json')),new=True)


if __name__=='__main__':main()
