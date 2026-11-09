"""One local sequential executor; every scientific child has its own guard."""
import os,sys,time,subprocess,json
from datetime import datetime
import psutil
from common import ROOT,OUT,STOP,MSK,read,sha,write
assert not (OUT/'arm_queue_start.json').exists(),'Do not create a duplicate executor'
gatepid=int(sys.argv[1]);gatebirth=float(sys.argv[2]);phasehash=sha(OUT/'execution_protocol.json')
write('arm_queue_start.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),
    waits_for=dict(pid=gatepid,birth=gatebirth),stages=['R134_spatial','R134_temporal','C34'],
    execution_sha256=phasehash,stop=STOP.isoformat(),started=datetime.now(MSK).isoformat()))

def checkpoint(stage,**kw):
    write('arm_queue_status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw),new=False)

def alive(pid,birth):
    try:return psutil.Process(pid).create_time()==birth and psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:return False

def stop_tree(proc):
    root=psutil.Process(proc.pid);owned=root.children(recursive=True)+[root]
    for child in reversed(owned):
        try:child.kill()
        except psutil.NoSuchProcess:pass
    gone,live=psutil.wait_procs(owned,timeout=10);assert not live

checkpoint('waiting_for_G7',pid=gatepid,birth=gatebirth)
while alive(gatepid,gatebirth):
    if datetime.now(MSK)>=STOP:raise SystemExit('Deadline while waiting')
    time.sleep(1)
assert read(OUT/'G7_complete.json')['passed'],'Predecessor ended without accepted completion'
completed=[]
for arm in ['R134_spatial','R134_temporal','C34']:
    assert sha(OUT/'execution_protocol.json')==phasehash
    if datetime.now(MSK)>=STOP:raise SystemExit('Scientific save/stop reached')
    marker=OUT/(arm+'_complete.json')
    if marker.exists():
        result=read(marker);assert result['passed'] and result['execution_sha256']==phasehash
        completed.append(arm);continue
    env=os.environ.copy()
    if arm=='C34':env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=''
    log=OUT/(arm+'.queue.log');err=OUT/(arm+'.queue.err.log')
    with log.open('x',encoding='utf8') as out,err.open('x',encoding='utf8') as error:
        proc=subprocess.Popen([sys.executable,'-u',str(OUT/'native_run.py'),arm],cwd=ROOT,env=env,
            stdout=out,stderr=error,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        birth=psutil.Process(proc.pid).create_time();began=time.monotonic()
        checkpoint('running',arm=arm,pid=proc.pid,birth=birth,completed=completed)
        while proc.poll() is None:
            if datetime.now(MSK)>=STOP or (arm=='C34' and time.monotonic()-began>3600):
                stop_tree(proc);checkpoint('stopped',arm=arm,reason='deadline_or_core_hour_safety_bound');raise SystemExit(124)
            time.sleep(1)
        code=proc.wait()
    if code!=0:
        checkpoint('failed',arm=arm,exit_code=code,completed=completed);raise SystemExit(code)
    result=read(marker);assert result['passed'] and result['execution_sha256']==phasehash
    completed.append(arm);checkpoint('stage_complete',arm=arm,completed=completed)
checkpoint('all_arms_complete',completed=completed)
print('All3 remaining arms completed',flush=True)
