"""Deferred numerical audit after the existing scientific executor is done."""
import os,sys,time,subprocess
from datetime import datetime
import psutil
from common import ROOT,OUT,STOP,MSK,read,sha,write
assert not (OUT/'boundary_queue_start.json').exists()
parent=read(OUT/'recovery1_queue_start.json');pid=parent['pid'];birth=parent['birth']
scripts=['register_boundary.py','boundary_audit.py'];sources={n:sha(OUT/n) for n in scripts}
write('boundary_queue_start.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),
    waits_for=dict(pid=pid,birth=birth),scripts=scripts,sources=sources,stop=STOP.isoformat()))
def checkpoint(stage,**kw):
    value=dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw)
    for attempt in range(30):
        try:write('boundary_queue_status.json',value,new=False);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
def alive():
    try:return psutil.Process(pid).create_time()==birth and psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:return False
checkpoint('waiting_for_native_results')
while alive():
    if datetime.now(MSK)>=STOP:raise SystemExit('Save/stop while waiting')
    time.sleep(1)
assert read(OUT/'recovery1_queue_status.json')['stage']=='complete'
for name in scripts:
    assert sha(OUT/name)==sources[name]
    if datetime.now(MSK)>=STOP:raise SystemExit('Save/stop before audit')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (OUT/(name+'.deferred.log')).open('x',encoding='utf8') as out,(OUT/(name+'.deferred.err.log')).open('x',encoding='utf8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        checkpoint('running',script=name,pid=proc.pid,birth=psutil.Process(proc.pid).create_time())
        code=proc.wait()
    if code!=0:checkpoint('failed',script=name,exit_code=code);raise SystemExit(code)
checkpoint('complete')
