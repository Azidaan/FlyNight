"""Скрытая цепочка проверки, CPU расчёта и аудита с учётом своих процессов."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_tau_half_20261104'

def write(name,data):
    p=OUT/name;t=p.with_suffix('.next');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

def main():
    deadline=datetime.fromisoformat(json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['hard_stop']).timestamp()
    me=psutil.Process();me.nice(psutil.IDLE_PRIORITY_CLASS);write('controller_launch.json',dict(pid=me.pid,created=me.create_time()))
    stages=[('checks','runs/flyvis_tau_half_20261104/check_catalog.py'),('infer','scripts/flyvis_tau_half_20261104.py'),('audit','scripts/audit_flyvis_tau_half_20261104.py')]
    reports=[];owned_all={}
    for stage,script in stages:
        owned={};reason=None;env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['Y1_EXECUTION_DEVICE']='cpu'
        with (OUT/(stage+'_stdout.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'_stderr.log')).open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            proc=psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS);owned[proc.pid]=proc.create_time();write(stage+'_launch.json',dict(pid=proc.pid,created=proc.create_time()))
            start=time.time()
            while child.poll() is None:
                try:
                    for q in proc.children(recursive=True):owned[q.pid]=q.create_time()
                except psutil.NoSuchProcess:pass
                status=OUT/'status.json';newest=max(start,status.stat().st_mtime if status.exists() else start)
                if time.time()>=deadline or (stage=='infer' and time.time()-newest>600):
                    reason='hard_stop' if time.time()>=deadline else 'watchdog'
                    for pid,birth in reversed(list(owned.items())):
                        try:
                            q=psutil.Process(pid)
                            if abs(q.create_time()-birth)<.01:q.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.5)
            child.wait(timeout=20)
        alive=[]
        for pid,birth in owned.items():
            try:
                if abs(psutil.Process(pid).create_time()-birth)<.01:alive.append(pid)
            except psutil.NoSuchProcess:pass
        report=dict(stage=stage,exit_code=child.returncode,reason=reason,owned=owned,all_owned_ended=not alive,remaining=alive,ended_utc=datetime.now(timezone.utc).isoformat())
        write(stage+'_controller_result.json',report);reports.append(report);owned_all.update(owned)
        if child.returncode or reason or alive:
            write('chain_result.json',dict(complete=False,failed_stage=stage,stages=reports,all_owned=owned_all));return 1
    write('chain_result.json',dict(complete=True,stages=reports,all_owned=owned_all,all_owned_ended=True));return 0
if __name__=='__main__':raise SystemExit(main())
