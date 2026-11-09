"""One replacement executor; failed original queues are retained."""
import os,sys,time,subprocess
from datetime import datetime
import psutil
from common import ROOT,OUT,STOP,MSK,read,sha,write
assert not (OUT/'recovery1_queue_start.json').exists(),'No duplicate executor'
phase=read(OUT/'status_recovery_protocol.json')
assert read(OUT/'status_retry_test.json')['passed']
stages=[('native_run.py',a) for a in ['R134_spatial','R134_temporal','C34']]
stages += [(n,None) for n in ['register_evaluation.py','deployment_audit.py','evaluate_native.py','native_statistics.py','id_fate.py']]
write('recovery1_queue_start.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),stages=stages,
    recovery_sha256=sha(OUT/'status_recovery_protocol.json'),stop=STOP.isoformat()))
def checkpoint(stage,**kw):
    value=dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw)
    for attempt in range(30):
        try:write('recovery1_queue_status.json',value,new=False);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
completed=[]
for target,arm in stages:
    if datetime.now(MSK)>=STOP:raise SystemExit('Scientific save/stop reached')
    label=arm or target
    env=os.environ.copy()
    if arm=='C34':env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=''
    command=[sys.executable,'-u',str(OUT/'status_write_recovery.py'),target]+([arm] if arm else [])
    with (OUT/(label+'.recovery1.log')).open('x',encoding='utf8') as out,(OUT/(label+'.recovery1.err.log')).open('x',encoding='utf8') as err:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        birth=psutil.Process(proc.pid).create_time();began=time.monotonic()
        checkpoint('running',target=target,arm=arm,pid=proc.pid,birth=birth,completed=completed)
        while proc.poll() is None:
            if datetime.now(MSK)>=STOP or (arm=='C34' and time.monotonic()-began>3600):
                root=psutil.Process(proc.pid);owned=root.children(recursive=True)+[root]
                for child in reversed(owned):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                gone,live=psutil.wait_procs(owned,timeout=10);assert not live
                checkpoint('stopped',target=target,reason='deadline');raise SystemExit(124)
            time.sleep(1)
        code=proc.wait()
    if code!=0:checkpoint('failed',target=target,arm=arm,exit_code=code,completed=completed);raise SystemExit(code)
    completed.append(label);checkpoint('stage_complete',target=target,arm=arm,completed=completed)
checkpoint('complete',completed=completed)
print('Recovery chain completed',flush=True)
