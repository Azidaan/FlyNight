"""Audit the completed parent only after the current guarded queue has ended."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
OWN=ROOT/'runs/singleton_verify_20261108'


def main():
    import psutil
    receipt=read(OWN/'audit_wait_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (OWN/'audit_queue_result.json').exists():
        if not process.is_running():raise RuntimeError('Own-template audit ended without its completion record; recover it first.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:process.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    sources={}
    for name in ['full_precision_execution_phase.json','full_statistics_phase.json','full_additional_phase.json']:
        for n,h in read(RUN/name)['sources'].items():assert sha(ROOT/n)==h;sources[n]=h
    for path in [Path(__file__),Path(__file__).with_name('audit_full_result.py'),ROOT/'research_singleton_free_20261108/full_audit.py',
        ROOT/'research_singleton_verify_20261108/common.py',RUN/'full_precision/comparison_report.json',RUN/'full_precision/additional_report.json',RUN/'full_precision/analyses_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'parent_full_audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only=True),new=True)
    folder=RUN/'parent_full_audit';folder.mkdir(exist_ok=True)
    with (folder/'audit.log').open('x',encoding='utf-8') as out,(folder/'audit.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/audit_full_result.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/'started.json',dict(pid=p.pid,created=psutil.Process(p.pid).create_time()),new=True);code=p.wait()
    lines=[line for line in (folder/'audit.log').read_text(encoding='utf-8').splitlines() if line.startswith('SYSTEM_MEMORY_GUARD ')]
    assert lines,(code,'No completed memory guardian');report=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            child=psutil.Process(int(pid))
            if abs(child.create_time()-birth)<.01:
                try:child.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/'ended.json',dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended
    assert read(RUN/'full_precision/independent_audit.json')['passed']


if __name__=='__main__':main()
