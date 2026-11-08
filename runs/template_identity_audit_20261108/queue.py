"""Wait for the required Windows memory reserve, then run one frozen audit."""
import json,os,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
deadline=datetime(2026,11,9,7,35,tzinfo=timezone(timedelta(hours=3))).timestamp()
owner=psutil.Process();owner.nice(psutil.IDLE_PRIORITY_CLASS)
def status(state,**extra):
    (OUT/'queue_status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),birth=owner.create_time(),updated=time.time(),deadline=deadline,**extra)),encoding='utf8')
status('starting')
with (OUT/'queue.log').open('x',encoding='utf8') as log:
    for step,ready in [('register','protocol.json'),('run','results.json')]:
        if (OUT/ready).exists():continue
        while True:
            memory=psutil.virtual_memory();required=max(6*2**30,int(.2*memory.total))+2*2**30
            status('waiting_for_memory',step=step,available_bytes=memory.available,required_start_bytes=required)
            if time.time()>=deadline:status('deadline_without_start',step=step);raise SystemExit(125)
            if memory.available>=required:break
            time.sleep(5)
        status('running',step=step)
        p=subprocess.Popen([sys.executable,'-u',str(OUT/'audit.py'),step],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        child=psutil.Process(p.pid);status('running',step=step,child_pid=p.pid,child_birth=child.create_time())
        code=p.wait();log.flush()
        if code:
            status('failed',step=step,returncode=code)
            raise SystemExit(code)
status('complete',result='results.json')
