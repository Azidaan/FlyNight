"""One sequential guarded CPU comparison from saved scalar features."""
import os,sys,json,time,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def status(stage,**kw):
    path=OUT/'status.json';temp=path.with_suffix('.pending');temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**kw),ensure_ascii=False,indent=2),encoding='utf8')
    for _ in range(30):
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
proc=psutil.Process();proc.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'executor.json').open('x',encoding='utf8') as f:json.dump(dict(pid=proc.pid,birth=proc.create_time(),save_stop=STOP.isoformat()),f)
status('starting')
try:
    for name in ['preflight.py','study.py','audit.py','statistics.py']:
        remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
        with (OUT/(name+'.log')).open('xb') as log,(OUT/(name+'.err.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);p=psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time())
            with (OUT/(name+'.started.json')).open('x',encoding='utf8') as f:json.dump(identity,f)
            status(name,child=identity);code=child.wait(timeout=remaining+30)
        with (OUT/(name+'.ended.json')).open('x',encoding='utf8') as f:json.dump(dict(**identity,returncode=code),f)
        assert code==0,(name,code)
    status('ready_for_review',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
