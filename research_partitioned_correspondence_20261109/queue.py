"""A bounded sequential CPU readout comparison; source maps and fits reused."""
import os,sys,json,subprocess,time
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
def run(label,name,args=()):
    remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
    with (OUT/(label+'.log')).open('xb') as log,(OUT/(label+'.err.log')).open('xb') as err:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/name),*args],cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);proc=psutil.Process(child.pid);identity=dict(pid=proc.pid,birth=proc.create_time())
        with (OUT/(label+'.started.json')).open('x',encoding='utf8') as f:json.dump(identity,f)
        status(label,child=identity);code=child.wait(timeout=remaining+30)
    with (OUT/(label+'.ended.json')).open('x',encoding='utf8') as f:json.dump(dict(**identity,returncode=code),f)
    assert code==0,(label,code)
proc=psutil.Process();proc.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'executor.json').open('x',encoding='utf8') as f:json.dump(dict(pid=proc.pid,birth=proc.create_time(),save_stop=STOP.isoformat()),f)
status('starting')
try:
    run('preflight','preflight.py');run('extract_local_parts','study.py',['extract'])
    if json.loads((OUT/'readiness.json').read_text(encoding='utf8'))['passed']:
        run('fit_local_parts','study.py',['fit']);run('independent_audit','audit.py');status('ready_for_review',goal_achieved=False)
    else:status('readiness_failed',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
