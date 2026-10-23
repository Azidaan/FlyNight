"""Follow verified recovery2, finalize parent and then start frozen stock controls."""
from pathlib import Path
from datetime import datetime
import sys,os,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_stock_controls_20261024 import STAGE,read,write,sha,now
from scripts.z13_matched_pipeline_recovery2_20261024 import REC


def main():
    import msvcrt
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    lock=(STAGE/'queue3.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ps=WindowsProcesses();deadline=datetime.fromisoformat(read(STAGE/'protocol.json')['save_at'])
    old=read(STAGE/'queue2.json');assert ps.creation(old['pid'])!=old['birth'] and (STAGE/'queue2_failure.json').exists()
    assert not (STAGE/'controller.json').exists()
    write(STAGE/'queue3.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),started=now().isoformat(),service_sha256=sha(Path(__file__)),execution_v3_sha256=sha(STAGE/'execution_v3.json'),parent_recovery_protocol_sha256=sha(REC/'protocol.json')))
    while not (REC/'controller_result.json').exists():
        if now()>=deadline:raise TimeoutError('Parent not complete before stock deadline')
        time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    from scripts.z13_record_matched_trackers_20261024 import ended
    ended(REC)
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    parent=STAGE.parent
    if not (parent/'finish_recorded.json').exists():execute('record_matched_recovery2',['scripts/z13_record_matched_trackers_20261024.py'])
    verify_and_graph('matched_recovery2')
    write(ROOT/'runs/z13_weekend_20261024/matched_recovery2.json',dict(at=now().isoformat(),complete=True,previous_failure_preserved=True))
    assert (deadline-now()).total_seconds()>900
    cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/z13_bounded_recovery_controller_20261022.py'),'--run',str(STAGE),'--script',str(ROOT/'scripts/z13_matched_stock_execution3_20261024.py')]
    code=subprocess.call(cmd,cwd=ROOT,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
    write(STAGE/'queue3_result.json',dict(exit_code=code,ended=now().isoformat()));return code


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:write(STAGE/'queue3_failure.json',dict(error=repr(e),at=now().isoformat()));raise
