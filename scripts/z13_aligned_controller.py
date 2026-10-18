"""One-GPU day-bounded worker, independent save/stop and finite recording."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_aligned_core_20261018'


def main():
    import psutil
    protocol=read(RUN/'protocol.json')
    for rel,digest in protocol['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    own={os.getpid()}|{p.pid for p in psutil.Process().parents()}
    for p in psutil.process_iter(['cmdline']):
        if p.pid not in own and any(Path(x).name in {'z13_aligned_core.py',Path(__file__).name} for x in p.info['cmdline'] or []):
            raise RuntimeError(f'Duplicate controller/worker {p.pid}')
    if (RUN/'controller.json').exists():
        raise RuntimeError('Existing run must be inspected; do not duplicate')
    soft=datetime.fromisoformat(protocol['save_stop_at']).timestamp()
    hard=datetime.fromisoformat(protocol['hard_stop_at']).timestamp()
    if time.time()>=soft:
        raise RuntimeError('No new computation after day save cutoff')
    command=[sys.executable,'-u',str(ROOT/'scripts/z13_aligned_core.py'),'--deadline',str(soft)]
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born=psutil.Process(child.pid).create_time()
        write(RUN/'controller.json',dict(pid=child.pid,create_time=born,command=command,
              started_at=datetime.now(timezone.utc).isoformat(),soft_deadline=protocol['save_stop_at'],
              hard_deadline=protocol['hard_stop_at'],device='single GPU',chat_automation=False))
        stop_written=False; timed_out=False
        while child.poll() is None:
            if time.time()>=soft and not stop_written:
                (RUN/'stop.request').write_text('Independent controller: save and stop before end of authorized day\n',encoding='utf-8')
                stop_written=True
            if time.time()>=hard:
                process=psutil.Process(child.pid)
                assert abs(process.create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=20); timed_out=True; break
            time.sleep(2)
    write(RUN/'controller_result.json',dict(exit_code=child.returncode,worker_ended=True,hard_timeout=timed_out,
          ended=datetime.now(timezone.utc).isoformat(),stop_request_written=stop_written))
    with (RUN/'postprocess.stdout.log').open('w',encoding='utf-8') as out,(RUN/'postprocess.stderr.log').open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/z13_record_next.py'),'--aligned','--publish'],cwd=ROOT,stdout=out,stderr=err)
    write(RUN/'postprocess_result.json',dict(exit_code=result.returncode,chat_calls=0))


if __name__=='__main__':
    main()
