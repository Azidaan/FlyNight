import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    root=read(RUN/'started.json');deadline=datetime(2026,11,8,13,55,tzinfo=MSK)
    while not (RUN/'metadata_stage_complete.json').exists():
        if datetime.now(MSK)>=deadline:return
        try:p=psutil.Process(root['pid']);assert abs(p.create_time()-root['created'])<.01
        except psutil.NoSuchProcess:raise RuntimeError('Native metadata queue ended; preserve its failed stage before any recovery.')
        time.sleep(5)
    from research_native_confirm_policy_20261108.controller import run
    run('data_inventory','research_native_confirm_policy_20261108/setup_data.py')
    write(RUN/'data_planner_result.json',dict(passed=True,registration_sha256=sha(RUN/'data_registration.json')),new=True)


if __name__=='__main__':main()
