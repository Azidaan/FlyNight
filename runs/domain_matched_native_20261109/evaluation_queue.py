"""Wait for the identified native queue; never duplicate it or a ready metric."""
import os,sys,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def read(p):return json.loads(p.read_text(encoding='utf8'))
def live(pid,birth):
    try:return psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:return False
def status(stage,**kw):
    path=OUT/'evaluation_queue_status.json';temp=path.with_suffix('.pending');temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**kw),ensure_ascii=False,indent=2),encoding='utf8')
    for _ in range(30):
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
def run(name,file):
    seconds=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert seconds>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2';command=[sys.executable,'-u',str(OUT/file)]
    with (OUT/(name+'.log')).open('xb') as f,(OUT/(name+'.err.log')).open('xb') as e:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW);actual=psutil.Process(child.pid)
        identity=dict(pid=actual.pid,birth=actual.create_time(),command=command);(OUT/(name+'.started.json')).write_text(json.dumps(identity),encoding='utf8');status(name,child=identity)
        try:code=child.wait(timeout=seconds+25)
        except subprocess.TimeoutExpired:
            if live(identity['pid'],identity['birth']):
                owned=[(p.pid,p.create_time()) for p in actual.children(recursive=True)]+[(actual.pid,actual.create_time())]
                for pid,birth in reversed(owned):
                    if live(pid,birth):psutil.Process(pid).kill()
            raise
    (OUT/(name+'.ended.json')).write_text(json.dumps(dict(identity,returncode=code)),encoding='utf8');assert code==0,(name,code)
self=psutil.Process();self.nice(psutil.IDLE_PRIORITY_CLASS);assert not (OUT/'evaluation_queue_identity.json').exists()
(OUT/'evaluation_queue_identity.json').write_text(json.dumps(dict(pid=self.pid,birth=self.create_time(),deadline=STOP.isoformat())),encoding='utf8');status('starting')
try:
    assert (OUT/'evaluation_protocol.json').exists();native=read(OUT/'queue_identity.json')
    while live(native['pid'],native['birth']):
        assert datetime.now(STOP.tzinfo)<STOP
        status('waiting_for_identified_native_queue',native=native);time.sleep(10)
    assert read(OUT/'queue_status.json')['stage']=='ready_for_evaluation','Native queue terminal without completed arms; do not repeat or evaluate partial trajectories'
    run('official_evaluation','evaluate_native.py');run('paired_statistics','native_statistics.py');run('direct_ID_fate','id_fate.py')
    status('results_ready_for_review',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
