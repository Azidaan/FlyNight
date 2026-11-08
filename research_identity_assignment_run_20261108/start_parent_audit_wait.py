import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    import psutil
    with (RUN/'parent_audit_wait.log').open('x',encoding='utf-8') as out,(RUN/'parent_audit_wait.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_identity_assignment_run_20261108/parent_audit_after_own.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_identity_assignment_run_20261108/parent_audit_after_own.py')
    write(RUN/'parent_audit_wait_started.json',root,new=True);roots=read(RUN/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(RUN/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)


if __name__=='__main__':main()
