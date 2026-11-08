"""Wait for the existing pilot, then run sequential first-problem gates only."""
import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def launch(name,script):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,59,30,tzinfo=MSK)
    folder=RUN/'children';folder.mkdir(parents=True,exist_ok=True);assert not (folder/(name+'.log')).exists()
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/'+script],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time()),new=True);code=proc.wait()
    reports=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')]
    assert reports,(name,code,'No completed memory guardian')
    report=json.loads(reports[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            process=psutil.Process(int(pid))
            if abs(process.create_time()-birth)<.01:
                try:process.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(name,code,report)


def main():
    import psutil
    phase();p=read(BASE/'full_precision/post_started.json');process=psutil.Process(p['pid']);assert abs(process.create_time()-p['created'])<.01
    while not (BASE/'full_precision/analyses_complete.json').exists():
        if not process.is_running():
            write(RUN/'waiting_stopped.json',dict(reason='Previous pilot postprocessor ended without a complete analysis manifest; recover it first.'),new=True);return
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    phase();status('first_problem_catalog_checks')
    launch('catalog','catalog_check.py');launch('technical','technical.py');launch('conditional','compare.py')
    report=read(RUN/'comparison_report.json')
    write(RUN/'queue_result.json',dict(passed=True,conditional_report_sha256=sha(RUN/'comparison_report.json'),
        full_tracker_gate=report['full_tracker_gate'],next='Register and execute full first-problem pilot if gate passes; do not start problems2–4.'),new=True)
    status('first_problem_conditional_complete',full_tracker_gate=report['full_tracker_gate'])


if __name__=='__main__':main()
