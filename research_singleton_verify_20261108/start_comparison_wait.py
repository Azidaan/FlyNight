import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    import psutil
    assert not (RUN/'comparison_wait_started.json').exists()
    # The observer only registers a stage already authorized by protocol;
    # the single data queue still dispatches the guarded heavy process.
    with (RUN/'comparison_wait.log').open('x',encoding='utf-8') as out,(RUN/'comparison_wait.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_verify_20261108/register_comparison.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_verify_20261108/register_comparison.py')
    write(RUN/'comparison_wait_started.json',receipt,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(receipt)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)


if __name__=='__main__':main()
