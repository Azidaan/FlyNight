"""Probe controller: one owned worker, 15-minute limit, no full evaluation."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    import msvcrt
    folder = RUN / 'technical'
    folder.mkdir(exist_ok=True)
    if (folder / 'status.json').exists():
        import shutil
        shutil.copy2(folder / 'status.json', folder / ('previous_status_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'))
    lock = (folder / 'controller.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
    began = time.monotonic()
    with (folder / 'log.txt').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u',
            'scripts/z13_spatial_hota_probe.py'], cwd=ROOT, creationflags=flags,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                 'OPENBLAS_NUM_THREADS': '2'}, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None and time.monotonic() - began < 900:
                atomic_json(folder / 'status.json', {'state': 'running', 'controller_pid': os.getpid(),
                    'child_pid': proc.pid, 'elapsed_seconds': time.monotonic() - began,
                    'limit_seconds': 900, 'updated': datetime.now().astimezone().isoformat()})
                time.sleep(5)
            if proc.poll() is None:
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    creationflags=subprocess.CREATE_NO_WINDOW, check=False, stdout=log, stderr=log)
                proc.wait(timeout=30)
                state = 'stopped_at_probe_limit'
            else:
                state = 'completed' if proc.returncode == 0 else 'failed'
            atomic_json(folder / 'status.json', {'state': state, 'exit_code': proc.returncode,
                'elapsed_seconds': time.monotonic() - began,
                'updated': datetime.now().astimezone().isoformat()})
        finally:
            if proc.poll() is None:
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    creationflags=subprocess.CREATE_NO_WINDOW, stdout=log, stderr=log)
            lock.close()


if __name__ == '__main__':
    main()
