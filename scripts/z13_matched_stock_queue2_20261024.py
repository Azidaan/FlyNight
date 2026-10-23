"""Follow corrected parent and start corrected remaining-price stock execution."""
from pathlib import Path
from datetime import datetime
import sys,os,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_stock_controls_20261024 import STAGE,read,write,sha,now
from scripts.z13_matched_pipeline_recovery_20261024 import REC


def main():
    import msvcrt
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    lock=(STAGE/'queue2.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();deadline=datetime.fromisoformat(read(STAGE/'protocol.json')['save_at'])
    assert not (STAGE/'controller.json').exists()
    write(STAGE/'queue2.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),service_sha256=sha(Path(__file__)),execution_v2_sha256=sha(STAGE/'execution_v2.json'),parent_recovery_protocol_sha256=sha(REC/'protocol.json'),science_protocol_sha256=sha(STAGE/'protocol.json')))
    while not (REC/'controller_result.json').exists():
        if now()>=deadline:raise TimeoutError('Parent not complete before stock deadline')
        time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    from scripts.z13_record_matched_trackers_20261024 import ended
    ended(REC);assert (deadline-now()).total_seconds()>900
    cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(STAGE),'--script',str(ROOT/'scripts/z13_matched_stock_execution2_20261024.py')]
    code=subprocess.call(cmd,cwd=ROOT,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);write(STAGE/'queue2_result.json',dict(exit_code=code,ended=now().isoformat()));return code


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:write(STAGE/'queue2_failure.json',dict(error=repr(e),at=now().isoformat()));raise
