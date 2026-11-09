"""Sequential local executor waiting on the already-live exact input recovery."""
import json,sys,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
CURRENT=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/c035e4ba95134d6f81ebeec929154a16'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def status(stage,**extra):
    path=OUT/'queue_status.json';tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(dict(stage=stage,updated=datetime.now(STOP.tzinfo).isoformat(),**extra),ensure_ascii=False,indent=2),encoding='utf8')
    for _ in range(30):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
def live(pid,birth):
    try:return psutil.Process(pid).create_time()==birth
    except psutil.NoSuchProcess:return False
self=psutil.Process();(OUT/'queue_identity.json').write_text(json.dumps(dict(pid=self.pid,birth=self.create_time(),deadline=STOP.isoformat())),encoding='utf8')
status('waiting_exact_input_recovery')
owner=read(CURRENT/'supervisor.json')
while live(owner['pid'],owner['birth']):
    if datetime.now(STOP.tzinfo)>=STOP:status('stopped_at_deadline');raise SystemExit(0)
    time.sleep(1)
guard=read(CURRENT/'memory_guard.json')
assert guard['reason'] is None and guard['returncode']==0,guard
assert read(OUT/'inputs_complete.json')['passed']
def run(name,script,args=(),gpu=False):
    remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else '';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
    log=OUT/(name+'.queue.log');error=OUT/(name+'.queue.err.log');assert not log.exists() and not error.exists()
    command=[sys.executable,'-u',str(OUT/script),*args]
    with log.open('xb') as f,error.open('xb') as e:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
        owned=psutil.Process(child.pid);identity=dict(pid=owned.pid,birth=owned.create_time(),command=command)
        (OUT/(name+'.started.json')).write_text(json.dumps(identity),encoding='utf8');status(name,child=identity)
        code=child.wait(timeout=remaining+30)
    (OUT/(name+'.ended.json')).write_text(json.dumps(dict(identity,returncode=code)),encoding='utf8')
    assert code==0,(name,code)
try:
    run('register_execution','compute.py',['register'])
    run('restore_missing_maps','compute.py',['restore'],gpu=True)
    run('extract_fixed_features','compute.py',['extract'])
    run('fit_fixed_heads','compute.py',['fit'])
    run('independent_audit','audit.py')
    if read(OUT/'summary.json')['gate']:run('final_training_fit','compute.py',['deploy'])
    run('record_result','record_result.py')
    run('check_records','check_records.py')
    status('ready_for_native_protocol' if read(OUT/'summary.json')['gate'] else 'reader_gate_failed_review_needed',goal_achieved=False)
except BaseException as exc:
    status('failed',error=repr(exc),goal_achieved=False);raise
