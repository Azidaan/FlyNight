"""Independent bounded CPU controller; kills only its own spawned worker tree on timeout."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_target_center_20261018'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--budget',type=int,required=True);a=ap.parse_args()
    stem='benchmark_controller' if a.benchmark else 'controller'
    result=RUN/(stem+'_result.json')
    if result.exists():raise RuntimeError('Controller result exists; do not overwrite')
    cmd=[sys.executable,'-u',str(ROOT/'scripts/z13_target_center_probe.py'),'--budget',str(a.budget)]
    if a.benchmark:cmd.append('--benchmark')
    with (RUN/(stem+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(stem+'.stderr.log')).open('w',encoding='utf-8') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        (RUN/(stem+'.json')).write_text(json.dumps(dict(pid=p.pid,command=cmd,budget_seconds=a.budget)),encoding='utf-8')
        t0=time.monotonic();hard=False
        try:code=p.wait(timeout=a.budget+30)
        except subprocess.TimeoutExpired:
            hard=True;subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],check=True,capture_output=True);code=p.wait(timeout=20)
        result.write_text(json.dumps(dict(exit_code=code,hard_timeout=hard,worker_ended=True,seconds=time.monotonic()-t0)),encoding='utf-8')
        if hard:(RUN/'status.json').write_text(json.dumps(dict(state='paused',stage='Hard CPU deadline; own worker ended',done=0,total=0)),encoding='utf-8')
        return code

if __name__=='__main__':sys.exit(main())
