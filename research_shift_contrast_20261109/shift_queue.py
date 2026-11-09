"""One sequential CPU gate, independent of the model's observation sessions."""
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
def run(name,file,args=()):
    remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2';command=[sys.executable,'-u',str(OUT/file),*args]
    with (OUT/(name+'.log')).open('xb') as f,(OUT/(name+'.err.log')).open('xb') as e:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW);proc=psutil.Process(child.pid);identity=dict(pid=proc.pid,birth=proc.create_time(),command=command)
        (OUT/(name+'.started.json')).write_text(json.dumps(identity),encoding='utf8');status(name,child=identity)
        code=child.wait(timeout=remaining+25)
    (OUT/(name+'.ended.json')).write_text(json.dumps(dict(identity,returncode=code)),encoding='utf8');assert code==0,(name,code)
self=psutil.Process();self.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'executor_identity.json').open('x',encoding='utf8') as f:json.dump(dict(pid=self.pid,birth=self.create_time(),deadline=STOP.isoformat()),f)
status('starting')
try:
    run('extract_missing_profiles','study.py',['extract']);run('fit_fixed_models','study.py',['fit']);run('independent_audit','audit.py')
    status('ready_for_review',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
