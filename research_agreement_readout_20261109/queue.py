"""One guarded sequential CPU experiment; no new maps, video or GPU calls."""
import os,sys,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def status(stage,**kw):
    path=OUT/'status.json';temp=path.with_suffix('.pending');temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**kw),indent=2),encoding='utf8')
    for _ in range(30):
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
p=psutil.Process();p.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'executor.json').open('x',encoding='utf8') as f:json.dump(dict(pid=p.pid,birth=p.create_time()),f)
status('starting')
try:
    for tag,args in [('preflight',['preflight.py']),('extract',['study.py','extract']),('fit',['study.py','fit']),('audit',['audit.py']),('statistics',['statistics.py'])]:
        assert datetime.now(STOP.tzinfo)<STOP
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
        with (OUT/(tag+'.log')).open('xb') as log,(OUT/(tag+'.err.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-u',str(OUT/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);proc=psutil.Process(child.pid);identity=dict(pid=proc.pid,birth=proc.create_time())
            with (OUT/(tag+'.started.json')).open('x',encoding='utf8') as f:json.dump(identity,f)
            status(tag,child=identity);code=child.wait(timeout=max(1,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
        with (OUT/(tag+'.ended.json')).open('x',encoding='utf8') as f:json.dump(dict(**identity,returncode=code),f)
        assert code==0,(tag,code)
    status('ready_for_review',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
