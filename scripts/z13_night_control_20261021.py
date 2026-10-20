"""Independent bounded controller for an explicitly named owned worker only."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import write
MSK=timezone(timedelta(hours=3))


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--script',type=Path,required=True);a.add_argument('--label',required=True);a.add_argument('--budget',type=int,required=True);x=a.parse_args()
    run=x.run.resolve();script=x.script.resolve();run.relative_to(ROOT);script.relative_to(ROOT)
    if not x.label.isidentifier() or not 0<x.budget<=14400:raise ValueError('Unsafe label/budget')
    marker=run/(x.label+'_controller_result.json')
    if marker.exists():raise RuntimeError('Existing attempt cannot be overwritten')
    now=lambda:datetime.now(MSK)
    soft=min(now()+timedelta(seconds=x.budget),datetime(2026,10,21,7,35,tzinfo=MSK))
    hard=min(soft+timedelta(seconds=20),datetime(2026,10,21,7,39,30,tzinfo=MSK))
    if now()>=soft:raise RuntimeError('No authorized work window')
    cmd=[sys.executable,'-u',str(script),'--worker','--attempt',x.label,'--budget',str(x.budget)]
    with (run/(x.label+'.stdout.log')).open('w',encoding='utf-8') as out,(run/(x.label+'.stderr.log')).open('w',encoding='utf-8') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        write(run/'controller.json',dict(worker_pid=child.pid,command=cmd,started_at=now().isoformat(),save_stop_at=soft.isoformat(),hard_stop_at=hard.isoformat(),phase=x.label))
        killed=False
        while child.poll() is None:
            if now()>=soft:(run/(x.label+'.stop')).touch(exist_ok=True)
            if now()>=hard:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True);killed=True;break
            time.sleep(.5)
        code=child.wait(timeout=15);write(marker,dict(worker_ended=True,exit_code=code,hard_timeout=killed,ended_at=now().isoformat()))
    return code


if __name__=='__main__':sys.exit(main())
