"""Одна полная версия FP64: контрольGPU, основнойCPU, независимыйаудит."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_y1_latency_fp64_20261104 import OUT,write,protocol


def main():
    p=protocol();deadline=datetime.fromisoformat(p['hard_stop']).timestamp()
    pre=json.loads((OUT/'preflight_checks.json').read_text());assert pre['passed'] and len(pre['checks'])==2
    me=psutil.Process();me.nice(psutil.IDLE_PRIORITY_CLASS)
    write('controller_launch.json',dict(pid=me.pid,created=me.create_time()))
    stages=[('gpu_reference','scripts/flyvis_y1_fp64_gpu_check_20261104.py',[]),
            ('infer','scripts/flyvis_y1_latency_fp64_20261104.py',['--stage','infer']),
            ('audit','scripts/audit_flyvis_y1_latency_fp64_20261104.py',[])]
    reports=[];all_owned={}
    for stage,script,extra in stages:
        env=dict(os.environ);env['Y1_EXECUTION_DEVICE']='cuda' if stage=='gpu_reference' else 'cpu'
        if stage!='gpu_reference':env['CUDA_VISIBLE_DEVICES']=''
        owned={};reason=None
        with (OUT/(stage+'_stdout.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'_stderr.log')).open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/script),*extra],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            proc=psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS);owned[proc.pid]=proc.create_time()
            write(stage+'_launch.json',dict(pid=proc.pid,created=proc.create_time()));started=time.time()
            while child.poll() is None:
                try:
                    for q in proc.children(recursive=True):owned[q.pid]=q.create_time()
                except psutil.NoSuchProcess:pass
                status=OUT/('gpu_reference/status.json' if stage=='gpu_reference' else 'status.json')
                newest=max(started,status.stat().st_mtime if status.exists() else started)
                if time.time()>=deadline or time.time()-newest>600:
                    reason='hard_stop' if time.time()>=deadline else 'watchdog_600s'
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
        result=dict(stage=stage,exit_code=child.returncode,stop_reason=reason,owned=owned,remaining=alive,
                    all_owned_ended=not alive,ended_utc=datetime.now(timezone.utc).isoformat())
        write(stage+'_controller_result.json',result);reports.append(result);all_owned.update(owned)
        if child.returncode or reason or alive:
            write('chain_result.json',dict(complete=False,failed_stage=stage,stages=reports,all_owned=all_owned));return 1
    write('chain_result.json',dict(complete=True,stages=reports,all_owned=all_owned,all_owned_ended=True));return 0


if __name__=='__main__':raise SystemExit(main())
