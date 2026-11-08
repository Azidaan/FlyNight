"""Bounded independent sequential verification queue, no new scientific scores."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
owner=psutil.Process();owner.nice(psutil.IDLE_PRIORITY_CLASS)
def status(value):
    (OUT/'check_status.json').write_text(json.dumps(dict(state=value,updated=time.time(),pid=os.getpid(),birth=owner.create_time())),encoding='utf8')
status('starting')
steps=[]
with (OUT/'check_queue.log').open('x',encoding='utf8') as log:
    for name in ['check_records.py','check_catalog.py','final_integrity.py']:
        status(name)
        result=subprocess.run([sys.executable,str(OUT/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        steps.append(dict(step=name,returncode=result.returncode))
        if result.returncode:
            status('failed')
            (OUT/'check_queue_result.json').write_text(json.dumps(dict(passed=False,steps=steps)),encoding='utf8')
            raise SystemExit(result.returncode)
status('complete')
(OUT/'check_queue_result.json').write_text(json.dumps(dict(passed=True,steps=steps)),encoding='utf8')
