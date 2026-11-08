"""Finish and verify this one audit after its existing queue completes."""
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];owner=psutil.Process();owner.nice(psutil.IDLE_PRIORITY_CLASS)
deadline=datetime(2026,11,9,7,35,tzinfo=timezone(timedelta(hours=3))).timestamp()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def status(state,**extra):(OUT/'report_queue_status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),birth=owner.create_time(),updated=time.time(),**extra)),encoding='utf8')
status('waiting_for_audit')
while True:
    s=read(OUT/'queue_status.json')
    if s['state']=='complete':break
    if s['state'] in ['failed','deadline_without_start']:status('audit_not_complete',audit=s);raise SystemExit(125)
    if not psutil.pid_exists(s['pid']) or psutil.Process(s['pid']).create_time()!=s['birth']:status('audit_executor_missing');raise SystemExit(125)
    if time.time()>=deadline:status('deadline');raise SystemExit(125)
    time.sleep(5)
steps=[]
with (OUT/'report_queue.log').open('x',encoding='utf8') as log:
    for name in ['finish.py','check_records.py','check_catalog.py','final_integrity.py']:
        while True:
            m=psutil.virtual_memory();required=max(6*2**30,int(.2*m.total))+2*2**30
            status('waiting_for_memory',step=name,available_bytes=m.available,required_bytes=required)
            if time.time()>=deadline:status('deadline',step=name);raise SystemExit(125)
            if m.available>=required:break
            time.sleep(5)
        p=subprocess.Popen([sys.executable,'-u',str(OUT/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        child=psutil.Process(p.pid);status('running',step=name,child_pid=p.pid,child_birth=child.create_time());code=p.wait();log.flush();steps.append(dict(step=name,returncode=code))
        if code:status('failed',step=name,returncode=code);raise SystemExit(code)
(OUT/'report_queue_result.json').write_text(json.dumps(dict(passed=True,steps=steps)),encoding='utf8');status('complete')
