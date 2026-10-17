"""Independent CPU recovery controller, separate logs and bounded own process tree."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/z13_target_center_20261018'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--budget',type=int,required=True);a=ap.parse_args()
    stem='recovery_benchmark' if a.benchmark else 'recovery_full'
    result=RUN/(stem+'_result.json')
    if result.exists():raise RuntimeError('Recovery result exists')
    protocol=json.loads((RUN/'protocol.json').read_text())
    for name,digest in protocol['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    cmd=[sys.executable,'-u',str(ROOT/'scripts/z13_target_center_cpu_recovery.py'),'--budget',str(a.budget)]
    if a.benchmark:cmd.append('--benchmark')
    with (RUN/(stem+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(stem+'.stderr.log')).open('w',encoding='utf-8') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        meta=dict(pid=p.pid,command=cmd,budget_seconds=a.budget,
                  technical_fix='Force CPU availability before FlyVis import; original scientific files unchanged',
                  files_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/z13_target_center_cpu_recovery.py','scripts/z13_target_center_recovery_controller.py']})
        (RUN/(stem+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
        t0=time.monotonic();hard=False
        try:code=p.wait(timeout=a.budget+30)
        except subprocess.TimeoutExpired:
            hard=True;subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],check=True,capture_output=True);code=p.wait(timeout=20)
        result.write_text(json.dumps(dict(exit_code=code,hard_timeout=hard,worker_ended=True,seconds=time.monotonic()-t0)),encoding='utf-8')
        if hard:(RUN/'status.json').write_text(json.dumps(dict(state='paused',stage='CPU recovery deadline; own worker ended',done=0,total=0)),encoding='utf-8')
        return code

if __name__=='__main__':sys.exit(main())
