"""Wait for the verified arm executor; then evaluate sequentially once."""
import os,sys,time,subprocess
from datetime import datetime
import psutil
from common import ROOT,OUT,STOP,MSK,read,sha,write
assert not (OUT/'evaluation_queue_start.json').exists()
parent=read(OUT/'arm_queue_start.json');pid=parent['pid'];birth=parent['birth']
scripts=['register_evaluation.py','deployment_audit.py','evaluate_native.py','native_statistics.py','id_fate.py']
sources={n:sha(OUT/n) for n in scripts}
write('evaluation_queue_start.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),
    waits_for=dict(pid=pid,birth=birth),scripts=scripts,sources=sources,stop=STOP.isoformat()))
def status(stage,**kw):write('evaluation_queue_status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw),new=False)
def alive():
    try:return psutil.Process(pid).create_time()==birth and psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:return False
status('waiting_for_arm_queue')
while alive():
    if datetime.now(MSK)>=STOP:raise SystemExit('Save/stop while waiting')
    time.sleep(1)
assert read(OUT/'arm_queue_status.json')['stage']=='all_arms_complete'
completed=[]
for name in scripts:
    assert sha(OUT/name)==sources[name]
    if datetime.now(MSK)>=STOP:raise SystemExit('Save/stop before stage')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (OUT/(name+'.queue.log')).open('x',encoding='utf8') as out,(OUT/(name+'.queue.err.log')).open('x',encoding='utf8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        child_birth=psutil.Process(proc.pid).create_time()
        status('running',script=name,pid=proc.pid,birth=child_birth,completed=completed)
        while proc.poll() is None:
            if datetime.now(MSK)>=STOP:
                root=psutil.Process(proc.pid);owned=root.children(recursive=True)+[root]
                for child in reversed(owned):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                gone,live=psutil.wait_procs(owned,timeout=10);assert not live
                status('deadline_stop',script=name);raise SystemExit(124)
            time.sleep(1)
        code=proc.wait()
    if code!=0:status('failed',script=name,exit_code=code,completed=completed);raise SystemExit(code)
    completed.append(name);status('stage_complete',script=name,completed=completed)
status('evaluation_complete',completed=completed)
print('Official metrics, inference and direct ID fate completed',flush=True)
