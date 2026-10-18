"""Finite independent controllers; no calls to chat or model automations."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import RUN, read, write, verify, status


def main():
    import psutil
    protocol = read(RUN / 'protocol.json'); verify(protocol)
    if (RUN / 'launch.json').exists():
        raise RuntimeError('Existing launch must be inspected, not duplicated')
    excluded = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    for process in psutil.process_iter(['cmdline']):
        if process.pid not in excluded and any(Path(x).name in
              {'z13_next_controller.py', 'z13_parallel_next.py'} for x in process.info['cmdline'] or []):
            raise RuntimeError(f'Duplicate process {process.pid}')
    jobs = {}
    for kind in ('utility', 'differentiable'):
        folder = RUN / kind; folder.mkdir(exist_ok=True)
        command = [sys.executable, '-u', str(ROOT / 'scripts/z13_parallel_next.py'), kind]
        start = time.time()
        with (folder / 'stdout.log').open('w', encoding='utf-8') as out, (folder / 'stderr.log').open('w', encoding='utf-8') as err:
            child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                     creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born = psutil.Process(child.pid).create_time()
        jobs[kind] = dict(child=child, born=born, start=start, folder=folder)
        write(folder / 'controller.json', dict(pid=child.pid, create_time=born, command=command,
              started_at=datetime.fromtimestamp(start, timezone.utc).isoformat(),
              hard_deadline=datetime.fromtimestamp(start + 610, timezone.utc).isoformat(), budget_seconds=600))
    write(RUN / 'launch.json', dict(controller_pid=os.getpid(), jobs={
        k: dict(pid=v['child'].pid, create_time=v['born']) for k, v in jobs.items()}, chat_automation=False))
    while jobs:
        for kind, job in list(jobs.items()):
            child = job['child']; timed_out = time.time() - job['start'] > 610
            if child.poll() is None and timed_out:
                process = psutil.Process(child.pid)
                assert abs(process.create_time() - job['born']) < .01
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=True, capture_output=True)
                child.wait(timeout=20)
                status(job['folder'], state='paused', stage='CPU deadline; partial outputs saved', done=0, total=0)
            if child.poll() is not None:
                write(job['folder'] / 'controller_result.json', dict(exit_code=child.returncode,
                      worker_ended=True, timed_out=timed_out, seconds=time.time()-job['start']))
                del jobs[kind]
        if jobs:
            time.sleep(1)
    write(RUN / 'controller_result.json', dict(worker_ended=True, jobs={k: read(RUN/k/'controller_result.json')
          for k in ('utility', 'differentiable')}, finished=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
