"""Attach an Idle audit observer to the existing author-window stop watch."""
import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    assert not (RUN/'full_audit_wait_started.json').exists()
    with (RUN/'full_audit_wait.log').open('x',encoding='utf-8') as out,(RUN/'full_audit_wait.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/full_audit_wait.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_free_20261108/full_audit_wait.py')
    write(RUN/'full_audit_wait_started.json',receipt,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(receipt)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'full_audit_wait_deadline_registered.json',dict(passed=True,root=receipt,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
