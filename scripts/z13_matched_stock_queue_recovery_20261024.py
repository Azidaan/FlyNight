"""Retain the stock queue failure and follow the corrected parent execution."""
from pathlib import Path
import sys,os,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_stock_controls_20261024 import STAGE,read,write,sha,now
from scripts.z13_matched_pipeline_recovery_20261024 import REC
from datetime import datetime


def main():
    import msvcrt
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    lock=(STAGE/'queue_recovery.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();p=read(STAGE/'protocol.json');deadline=datetime.fromisoformat(p['save_at'])
    write(STAGE/'queue_recovery.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),service_sha256=sha(Path(__file__)),parent_recovery_protocol_sha256=sha(REC/'protocol.json'),old_queue_failure_sha256=sha(STAGE/'queue_failure.json'),science_protocol_sha256=sha(STAGE/'protocol.json'),reason='Original parent stopped at incorrect remaining-job price accounting. Original queue failure retained. Wait for corrected parent within its unchanged original budget. Stock science and budget unchanged.'))
    while not (REC/'controller_result.json').exists():
        if now()>=deadline:raise TimeoutError('Corrected parent not complete before stock deadline')
        time.sleep(5)
    parent=read(REC/'controller.json')
    while ps.creation(parent['pid'])==parent['birth']:
        if now()>=deadline:raise TimeoutError('Parent controller still alive')
        time.sleep(.5)
    from scripts.z13_record_matched_trackers_20261024 import ended
    ended(REC);assert (deadline-now()).total_seconds()>900
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(STAGE),'--script',str(ROOT/'scripts/z13_matched_stock_controls_20261024.py')]
    code=subprocess.call(command,cwd=ROOT,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    write(STAGE/'queue_recovery_result.json',dict(exit_code=code,ended=now().isoformat()));return code


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:write(STAGE/'queue_recovery_failure.json',dict(error=repr(e),at=now().isoformat()));raise
