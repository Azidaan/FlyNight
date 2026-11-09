"""Only the unfinished output analysis and numerical audit; no new inference."""
import os,sys,time,subprocess
from datetime import datetime
import psutil
from common import ROOT,OUT,STOP,MSK,read,sha,write
assert not (OUT/'recovery3_queue_start.json').exists()
assert read(OUT/'provenance_math_test.json')['passed']
scripts=['id_fate_v2.py','register_boundary.py','boundary_audit.py']
sources={n:sha(OUT/n) for n in scripts}
write('recovery3_queue_start.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),scripts=scripts,
    sources=sources,stop=STOP.isoformat()))
def checkpoint(stage,**kw):
    value=dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw)
    for attempt in range(30):
        try:write('recovery3_queue_status.json',value,new=False);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)
completed=[]
for name in scripts:
    assert sha(OUT/name)==sources[name]
    if datetime.now(MSK)>=STOP:raise SystemExit('Save/stop before analysis')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (OUT/(name+'.recovery3.log')).open('x',encoding='utf8') as out,(OUT/(name+'.recovery3.err.log')).open('x',encoding='utf8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        checkpoint('running',script=name,pid=proc.pid,birth=psutil.Process(proc.pid).create_time(),completed=completed)
        code=proc.wait()
    if code!=0:checkpoint('failed',script=name,exit_code=code,completed=completed);raise SystemExit(code)
    completed.append(name);checkpoint('stage_complete',script=name,completed=completed)
checkpoint('complete',completed=completed)
