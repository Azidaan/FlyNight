"""One hidden sequential executor; each heavy child has its own memory guard."""
import os,sys,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def read(p):return json.loads(p.read_text(encoding='utf8'))
def status(stage,**extra):
    path=OUT/'queue_status.json';temp=path.with_suffix('.pending')
    temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**extra),ensure_ascii=False,indent=2),encoding='utf8')
    for _ in range(30):
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
def live(pid,birth):
    try:return psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:return False
def run(name,file,args=(),gpu=False):
    seconds=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert seconds>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
    command=[sys.executable,'-u',str(OUT/file),*args]
    with (OUT/(name+'.log')).open('xb') as log,(OUT/(name+'.err.log')).open('xb') as err:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        proc=psutil.Process(child.pid);identity=dict(pid=proc.pid,birth=proc.create_time(),command=command)
        (OUT/(name+'.started.json')).write_text(json.dumps(identity),encoding='utf8');status(name,child=identity)
        try:code=child.wait(timeout=seconds+25)
        except subprocess.TimeoutExpired:
            # Children normally stop independently at STOP. If that fails,
            # terminate only this still-identified launch tree before hardstop.
            if live(identity['pid'],identity['birth']):
                owned=[(p.pid,p.create_time()) for p in proc.children(recursive=True)]+[(proc.pid,proc.create_time())]
                for pid,birth in reversed(owned):
                    if live(pid,birth):psutil.Process(pid).kill()
            raise
    (OUT/(name+'.ended.json')).write_text(json.dumps(dict(identity,returncode=code)),encoding='utf8')
    assert code==0,(name,code)
self=psutil.Process();self.nice(psutil.IDLE_PRIORITY_CLASS)
assert not (OUT/'queue_identity.json').exists()
(OUT/'queue_identity.json').write_text(json.dumps(dict(pid=self.pid,birth=self.create_time(),deadline=STOP.isoformat())),encoding='utf8')
status('starting',goal_achieved=False)
try:
    assert (OUT/'execution_protocol.json').exists()
    run('prepare_cache','prepare_cache.py')
    run('zero_witnesses','native_run.py',['zero'])
    run('prerun_checks','check_records.py')
    for arm in read(OUT/'protocol.json')['arms']:run('native_'+arm,'native_run.py',[arm],gpu=arm=='C34')
    status('ready_for_evaluation',goal_achieved=False,all_five_native_arms_complete=True)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
