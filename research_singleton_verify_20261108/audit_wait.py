import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    import psutil
    r=read(RUN/'queue_started.json');p=psutil.Process(r['pid']);assert abs(p.create_time()-r['created'])<.01
    while not (RUN/'data_queue_result.json').exists():
        if not p.is_running():raise RuntimeError('The own-template queue ended before its result; recover the exact error.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:p.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    sources=read(RUN/'comparison_phase.json')['sources'].copy()
    for path in [Path(__file__),Path(__file__).with_name('audit.py'),ROOT/'research_singleton_free_20261108/audit.py',RUN/'comparison_report.json',RUN/'selection.json',RUN/'scores_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)
    from research_singleton_verify_20261108.queue_data import launch
    launch('independent_audit','audit.py');assert read(RUN/'independent_audit.json')['passed']
    write(RUN/'audit_queue_result.json',dict(passed=True,audit_sha256=sha(RUN/'independent_audit.json')),new=True)


if __name__=='__main__':main()
