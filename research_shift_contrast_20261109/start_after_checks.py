"""Sequential checked launch; the frozen scientific queue is kept unchanged."""
import os,sys,json,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
self=psutil.Process();self.nice(psutil.IDLE_PRIORITY_CLASS)
with (OUT/'launch_chain.json').open('x',encoding='utf8') as f:json.dump(dict(pid=self.pid,birth=self.create_time(),save_stop=STOP.isoformat(),sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'preflight.py',OUT/'shift_queue.py',OUT/'protocol.json']}),f,indent=2)
for name in ['preflight.py','shift_queue.py']:
    remaining=(STOP-datetime.now(STOP.tzinfo)).total_seconds();assert remaining>0
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='2'
    with (OUT/(name+'.launcher.log')).open('xb') as log,(OUT/(name+'.launcher.err.log')).open('xb') as err:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,env=env,stdout=log,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        process=psutil.Process(child.pid)
        with (OUT/(name+'.launch.json')).open('x',encoding='utf8') as f:json.dump(dict(pid=process.pid,birth=process.create_time()),f)
        code=child.wait(timeout=remaining+30)
    with (OUT/(name+'.terminal.json')).open('x',encoding='utf8') as f:json.dump(dict(returncode=code),f)
    assert code==0,(name,code)
