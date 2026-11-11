"""Versioned import-isolation recovery; keep original failed logs untouched."""
from common import *
import subprocess
import psutil
import time

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
old=read(ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/c59e17a95d0d41aa8bad68d68dffe853/memory_guard.json')
assert old['returncode']==1 and old['reason'] is None
for pid,birth in old['owned'].items():
    try:assert psutil.Process(int(pid)).create_time()!=birth
    except psutil.NoSuchProcess:pass
verify();me=psutil.Process()
write('recovery1.json',dict(pid=me.pid,birth=me.create_time(),created=datetime.now(MSK).isoformat(),
    original_failure='torch import of queue resolved to task queue.py; no network calls or maps completed',
    unchanged_scientific_protocol_sha256=sha(OUT/'protocol.json'),old_owned_processes_ended=True,
    sources={n:sha(OUT/n) for n in ['launch_stage.py','resume_recovery1.py','recover.py','compare.py','audit.py']},
    change='Strip task folder from sys.path, explicitly load frozen common module, retain wrapper in supervisor re-entry.'))
try:
    for stage,proof in [('recover.py','maps_complete.json'),('compare.py','predictions.json'),('audit.py','summary.json')]:
        verify()
        if (OUT/proof).exists():assert read(OUT/proof)['passed'];continue
        with (OUT/(stage+'.recovery1.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch_stage.py'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            proc=psutil.Process(p.pid)
            write(stage+'.recovery1.started.json',dict(pid=p.pid,birth=proc.create_time(),started=datetime.now(MSK).isoformat()))
            while p.poll() is None:
                assert datetime.now(MSK)<STOP
                time.sleep(.5)
        write(stage+'.recovery1.ended.json',dict(returncode=p.returncode,ended=datetime.now(MSK).isoformat()))
        assert p.returncode==0 and read(OUT/proof)['passed'],stage
    status('completed',completed=3,total=3,summary='summary.json')
except BaseException as e:status('failed_recovery1',reason=repr(e));raise
