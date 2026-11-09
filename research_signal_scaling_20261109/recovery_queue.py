"""Sequential continuation after the recorded numerical stop; reuse passed QA."""
import os,sys,json,time,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def status(stage,**kw):
    path=OUT/'recovery_status.json';temp=path.with_suffix('.pending');temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**kw),indent=2),encoding='utf8');temp.replace(path)
p=psutil.Process();p.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'recovery_executor.json').open('x',encoding='utf8') as f:json.dump(dict(pid=p.pid,birth=p.create_time()),f)
assert json.loads((OUT/'preflight.json').read_text())['passed']
status('starting')
try:
    for name in ['recover_numerics.py','audit.py','statistics.py']:
        assert datetime.now(STOP.tzinfo)<STOP
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
        with (OUT/('recovery_'+name+'.log')).open('xb') as log,(OUT/('recovery_'+name+'.err.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);proc=psutil.Process(child.pid);identity=dict(pid=proc.pid,birth=proc.create_time())
            with (OUT/('recovery_'+name+'.started.json')).open('x',encoding='utf8') as f:json.dump(identity,f)
            status(name,child=identity);code=child.wait(timeout=max(1,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
        with (OUT/('recovery_'+name+'.ended.json')).open('x',encoding='utf8') as f:json.dump(dict(**identity,returncode=code),f)
        assert code==0,(name,code)
    status('ready_for_review',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
