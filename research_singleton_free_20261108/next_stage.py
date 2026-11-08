"""Wait for the exact conditional executor, audit, then dispatch only its fixed gate."""
import sys,time,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    receipt=read(RUN/'queue_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (RUN/'queue_result.json').exists():
        if not process.is_running():
            write(RUN/'next_stage_stopped.json',dict(reason='Conditional queue ended before its result. Preserve failure and recover the exact stage.'),new=True);return
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    phase();source=read(RUN/'phase.json')['sources'].copy()
    for path in [Path(__file__),ROOT/'research_singleton_free_20261108/audit.py',RUN/'comparison_report.json',RUN/'selection.json',RUN/'technical.json']:
        source[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=source),new=True)
    from research_singleton_free_20261108.queue import launch
    launch('independent_audit','audit.py');assert read(RUN/'independent_audit.json')['passed']
    report=read(RUN/'comparison_report.json')
    if report['full_tracker_gate']['proceed']:
        with (RUN/'register_full.log').open('x',encoding='utf-8') as out,(RUN/'register_full.err').open('x',encoding='utf-8') as err:
            result=subprocess.run([sys.executable,'-u','research_singleton_free_20261108/register_full.py'],cwd=ROOT,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        assert result.returncode==0
        write(RUN/'next_stage_result.json',dict(passed=True,full_pilot_dispatched=True,full_phase_sha256=sha(RUN/'full_phase.json')),new=True)
    else:
        write(RUN/'next_stage_result.json',dict(passed=True,full_pilot_dispatched=False,reason='Fixed screening criterion failed; no full new rollout and no other problem started.'),new=True)
        status('first_problem_screening_failed',next='Replace or stop this version; first problem not marked solved.')


if __name__=='__main__':main()
