"""Dispatch and register the first-problem queue with the existing author-window watcher."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    phase();status('waiting_for_existing_full_pilot',deadline='2026-11-08T14:00:00+03:00')
    with (RUN/'queue.log').open('x',encoding='utf-8') as out,(RUN/'queue.err').open('x',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/queue.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time(),script='research_singleton_free_20261108/queue.py')
    write(RUN/'queue_started.json',receipt,new=True)
    path=BASE/'deadline_roots.json';roots=read(path);roots['roots'].append(receipt);write(path,roots)
    print(proc.pid)


if __name__=='__main__':main()
