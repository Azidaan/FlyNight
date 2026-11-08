import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    root=read(RUN/'started.json')
    while not (RUN/'comparison_stage_complete.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        try:p=psutil.Process(root['pid']);assert abs(p.create_time()-root['created'])<.01
        except psutil.NoSuchProcess:raise RuntimeError('Causal comparison ended before its independent audit. Preserve failed stage before graph update.')
        time.sleep(5)
    from research_native_confirm_policy_20261108.controller import run
    run('graph_bookkeeping','research_singleton_admission_20261108/rebuild_graph.py')


if __name__=='__main__':main()
