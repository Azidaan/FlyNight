"""Resume only the corrected zero comparator and the unfinished comparison."""
import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_recovery2_20261108.bootstrap import install
from research_singleton_free_20261108.common import *


def main():
    import psutil
    install();phase();phase('recovery3_phase.json');began=time.monotonic()
    while not (RUN/'recovery3_deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'recovery3_deadline_registered.json').exists()
    from research_singleton_free_20261108.queue import launch
    launch('technical_recovery3','../research_singleton_free_recovery3_20261108/technical.py');launch('conditional','compare.py')
    sources=read(RUN/'phase.json')['sources'].copy()
    for path in [ROOT/'research_singleton_free_20261108/audit.py',RUN/'comparison_report.json',RUN/'selection.json',RUN/'technical.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    launch('independent_audit','audit.py');assert read(RUN/'independent_audit.json')['passed']
    report=read(RUN/'comparison_report.json')
    if not report['full_tracker_gate']['proceed']:
        write(RUN/'recovery3_result.json',dict(passed=True,full_dispatched=False,reason='Fixed diagnostic gate failed; first problem remains open.'),new=True)
        status('first_problem_screening_failed');return
    with (RUN/'register_full.log').open('x',encoding='utf-8') as out,(RUN/'register_full.err').open('x',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,'-u','research_singleton_free_20261108/register_full.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    assert result.returncode==0
    full=RUN/'full';receipt=read(full/'manager_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (full/'analyses_complete.json').exists():
        if not process.is_running():raise RuntimeError('Full manager stopped before complete analyses; preserve and recover the exact failure.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    from research_singleton_free_20261108.full_manager import launch as full_launch
    full_launch('independent_audit','full_audit.py',[]);assert read(full/'independent_audit.json')['passed']
    write(RUN/'recovery3_result.json',dict(passed=True,full_dispatched=True,audit_sha256=sha(full/'independent_audit.json')),new=True)


if __name__=='__main__':main()
