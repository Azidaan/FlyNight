"""Attach the bounded next-stage executor to the existing author-window watch."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    phase()
    with (RUN/'next_stage.log').open('x',encoding='utf-8') as out,(RUN/'next_stage.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/next_stage.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),script='research_singleton_free_20261108/next_stage.py')
    write(RUN/'next_stage_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root);write(BASE/'deadline_roots.json',roots);print(proc.pid)


if __name__=='__main__':main()
