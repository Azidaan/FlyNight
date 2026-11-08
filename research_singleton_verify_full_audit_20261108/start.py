import sys,time,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    assert not (RUN/'full_audit_wait_started.json').exists()
    with (RUN/'full_audit_wait.log').open('x',encoding='utf-8') as out,(RUN/'full_audit_wait.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_audit_20261108/wait_and_audit.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_singleton_verify_full_audit_20261108/wait_and_audit.py')
    write(RUN/'full_audit_wait_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'full_audit_deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
