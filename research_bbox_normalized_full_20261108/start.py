import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main():
    import psutil
    from research_bbox_normalized_full_20261108.register import main as register
    register()
    with (FULL/'controller.log').open('x',encoding='utf-8') as out,(FULL/'controller.err').open('x',encoding='utf-8') as err:child=subprocess.Popen([sys.executable,'-u','research_bbox_normalized_full_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_bbox_normalized_full_20261108/controller.py');write(FULL/'started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(FULL/'deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True);print(root)


if __name__=='__main__':main()
