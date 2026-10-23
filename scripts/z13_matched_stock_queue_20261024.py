"""No polling automation: one local queue process, no model calls or GPU use."""
from pathlib import Path
import sys,time,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_stock_controls_20261024 import STAGE,P,read,write,now,ROOT
from datetime import datetime


def main():
    import msvcrt
    lock=(STAGE/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();write(STAGE/'queue.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat()))
    p=read(STAGE/'protocol.json');deadline=datetime.fromisoformat(p['save_at'])
    while not (P.STAGE/'controller_result.json').exists():
        if now()>=deadline:raise TimeoutError('Parent not complete before stock-control deadline')
        time.sleep(5)
    # Controller writes result just before exiting. Verify identity, not stale PID.
    from scripts.z13_record_matched_trackers_20261024 import ended
    parent=read(P.STAGE/'controller.json')
    while ps.creation(parent['pid'])==parent['birth']:
        if now()>=deadline:raise TimeoutError('Parent controller still alive at deadline')
        time.sleep(.5)
    ended(P.STAGE)
    assert (deadline-now()).total_seconds()>900,'Insufficient time even for fixed technical price'
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(STAGE),'--script',str(ROOT/'scripts/z13_matched_stock_controls_20261024.py')]
    code=subprocess.call(command,cwd=ROOT,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    write(STAGE/'queue_result.json',dict(exit_code=code,ended=now().isoformat()))
    return code


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:write(STAGE/'queue_failure.json',dict(error=repr(e),at=now().isoformat()));raise
