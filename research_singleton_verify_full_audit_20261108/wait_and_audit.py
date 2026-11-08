"""Dispatch one source-bound audit after the exact full executor has exited."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main():
    import psutil
    began=time.monotonic()
    while not (RUN/'full_audit_deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'full_audit_deadline_registered.json').exists()
    receipt=read(FULL/'manager_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (FULL/'analyses_complete.json').exists():
        if not process.is_running():raise RuntimeError('Full manager ended before analyses completion; preserve and recover its failure.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    sources=read(RUN/'full_phase.json')['sources'].copy()
    for path in [Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'research_singleton_free_20261108/full_audit.py',
        FULL/'comparison_report.json',FULL/'additional_report.json',FULL/'analyses_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['trackers/byte_tracker.py','trackers/bot_sort.py','trackers/utils/stracks.py','trackers/utils/matching.py','utils/ops.py','utils/metrics.py']:
        path=ROOT/'.venv/Lib/site-packages/ultralytics'/name;sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only=True),new=True)
    folder=FULL/'audit';folder.mkdir(exist_ok=True)
    with (folder/'audit.log').open('x',encoding='utf-8') as out,(folder/'audit.err').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_audit_20261108/audit.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/'started.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),new=True);code=child.wait()
    lines=[s for s in (folder/'audit.log').read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')]
    assert lines,(code,'No completed memory guardian');report=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/'ended.json',dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended
    assert read(FULL/'independent_audit.json')['passed']
    write(RUN/'full_audit_queue_result.json',dict(passed=True,audit_sha256=sha(FULL/'independent_audit.json')),new=True)


if __name__=='__main__':main()
