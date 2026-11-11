"""Hidden sequential execution; every worker has its own memory/deadline guard."""
from common import *
import psutil
import subprocess
import time

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
me=psutil.Process(); write('executor.json',dict(pid=me.pid,birth=me.create_time(),deadline=STOP.isoformat(),
    script_sha256=sha(Path(__file__)),worker_sources={n:sha(OUT/n) for n in ['recover.py','compare.py','audit.py','common.py']}))
status('queued',completed=0,total=3)
try:
    gate = ROOT / read(OUT/'preflight_guard.json')['report']
    while not gate.exists() or read(gate).get('returncode') is None:
        assert datetime.now(MSK)<STOP
        time.sleep(1)
    assert read(gate)['returncode']==0 and read(gate)['reason'] is None
    assert read(OUT/'record_checks.json')['passed']
    for index,(script,proof) in enumerate([('recover.py','maps_complete.json'),('compare.py','predictions.json'),('audit.py','summary.json')]):
        verify()
        if (OUT/proof).exists():
            assert read(OUT/proof)['passed'];continue
        assert datetime.now(MSK)<STOP
        with (OUT/(script+'.log')).open('wb') as log:
            worker=subprocess.Popen([sys.executable,'-u',str(OUT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            proc=psutil.Process(worker.pid); birth=proc.create_time()
            write(script+'.started.json',dict(pid=worker.pid,birth=birth,script_sha256=sha(OUT/script),started=datetime.now(MSK).isoformat()))
            while worker.poll() is None:
                # The stage's independent supervisor also enforces this deadline.
                if datetime.now(MSK)>=STOP:
                    assert proc.create_time()==birth
                    children=proc.children(recursive=True)
                    for child in reversed(children):
                        try:child.terminate()
                        except psutil.NoSuchProcess:pass
                    proc.terminate();raise RuntimeError('Authorized deadline reached')
                time.sleep(.5)
        write(script+'.ended.json',dict(returncode=worker.returncode,ended=datetime.now(MSK).isoformat(),output_exists=(OUT/proof).exists()))
        assert worker.returncode==0 and read(OUT/proof)['passed'],script
    status('completed',completed=3,total=3,summary='summary.json')
except BaseException as exc:
    status('failed',reason=repr(exc));raise
