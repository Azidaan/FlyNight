"""Independent scoped stage controller with Moscow deadline and PID identity checks."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write
from scripts.z13_hota_deadline_guard import WindowsProcesses
MSK=timezone(timedelta(hours=3))


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--script',type=Path,required=True);x=a.parse_args()
    run=x.run.resolve();run.relative_to(ROOT.resolve());script=x.script.resolve();script.relative_to(ROOT.resolve())
    p=read(run/'protocol.json');now=lambda:datetime.now(MSK)
    if (run/'controller_result.json').exists():raise RuntimeError('Previous result exists; no duplicate')
    import msvcrt
    lock=(run/'controller.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    soft=min(now()+timedelta(seconds=p['budget_seconds']),datetime.fromisoformat(p['save_at']))
    hard=min(soft+timedelta(seconds=20),datetime.fromisoformat(p['hard_stop']))
    if now()>=soft:raise RuntimeError('No author work window')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'' if p['device']=='cpu' else '0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','GIT_OPTIONAL_LOCKS':'0'}
    processes=WindowsProcesses();known={};killed=False
    with (run/'worker.stdout.log').open('w',encoding='utf-8') as out,(run/'worker.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(script),'--worker'],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        birth=processes.creation(child.pid);assert birth is not None;known[child.pid]=birth
        write(run/'controller.json',dict(pid=os.getpid(),birth=processes.creation(os.getpid()),worker_pid=child.pid,worker_birth=birth,started=now().isoformat(),save_at=soft.isoformat(),hard_stop=hard.isoformat()))
        while child.poll() is None:
            live={pid for pid,b in known.items() if processes.creation(pid)==b};changed=True;rows=processes.children()
            while changed:
                changed=False
                for pid,parent in rows:
                    if parent in live and pid not in live:
                        b=processes.creation(pid)
                        if b is not None and b>=known[parent]:known[pid]=b;live.add(pid);changed=True
            if now()>=soft:(run/'stop.request').touch(exist_ok=True)
            if now()>=hard:
                for pid,b in known.items():
                    if processes.creation(pid)==b:subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,check=False,creationflags=subprocess.CREATE_NO_WINDOW)
                killed=True;break
            time.sleep(.5)
        code=child.wait(timeout=15)
        remaining={str(pid):b for pid,b in known.items() if processes.creation(pid)==b}
        write(run/'controller_result.json',dict(exit_code=code,hard_timeout=killed,worker_ended=child.poll() is not None,all_owned_ended=not remaining,remaining=remaining,owned={str(pid):b for pid,b in known.items()},ended=now().isoformat()))
        if remaining:raise RuntimeError('Owned child still alive')
    return code


if __name__=='__main__':sys.exit(main())
