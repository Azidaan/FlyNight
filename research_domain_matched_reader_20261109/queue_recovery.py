"""Resume the failed import stage; preserve the original queue and ready inputs."""
import os,sys,json,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def read(p):return json.loads(p.read_text(encoding='utf8'))
def status(stage,**extra):
    path=OUT/'queue_recovery_status.json';temp=path.with_suffix('.pending')
    temp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**extra),ensure_ascii=False,indent=2),encoding='utf8')
    for _ in range(30):
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
def live(pid,birth):
    try:return psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:return False
failed=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/2e62bf53d9f6423485d946ca95add11d'
old=read(OUT/'queue_identity.json');assert not live(old['pid'],old['birth'])
guard=read(failed/'memory_guard.json');assert guard['reason'] is None and guard['returncode']==1
assert all(not live(pid,birth) for pid,birth in guard['owned'].items())
assert read(OUT/'inputs_complete.json')['passed'] and not (OUT/'input_math_witness.json').exists()
passport=read(OUT/'import_recovery_protocol.json')
for name,digest in passport['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
self=psutil.Process();(OUT/'active_executor.json').write_text(json.dumps(dict(pid=self.pid,birth=self.create_time(),status='queue_recovery_status.json',deadline=STOP.isoformat(),previous='queue_identity.json')),encoding='utf8')
def run(name,target,args=(),gpu=False):
    remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
    name+='.recovery1';log=OUT/(name+'.log');error=OUT/(name+'.err.log');assert not log.exists() and not error.exists()
    command=[sys.executable,'-u',str(OUT/'safe_phase.py'),target,*args]
    with log.open('xb') as f,error.open('xb') as e:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
        actual=psutil.Process(child.pid);identity=dict(pid=actual.pid,birth=actual.create_time(),command=command)
        (OUT/(name+'.started.json')).write_text(json.dumps(identity),encoding='utf8');status(name,child=identity)
        code=child.wait(timeout=remaining+30)
    (OUT/(name+'.ended.json')).write_text(json.dumps(dict(identity,returncode=code)),encoding='utf8')
    assert code==0,(name,code)
try:
    run('restore_missing_maps','compute.py',['restore'],gpu=True)
    run('extract_fixed_features','compute.py',['extract'])
    run('fit_fixed_heads','compute.py',['fit'])
    run('independent_audit','audit.py')
    if read(OUT/'summary.json')['gate']:run('final_training_fit','compute.py',['deploy'])
    run('record_result','record_result.py')
    run('check_records','check_records.py')
    status('ready_for_native_protocol' if read(OUT/'summary.json')['gate'] else 'reader_gate_failed_review_needed',goal_achieved=False)
except BaseException as exc:status('failed',error=repr(exc),goal_achieved=False);raise
