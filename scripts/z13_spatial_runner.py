"""Persistent sequential controller. Own children only; no dev or heldout evaluation."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json

RUN = ROOT / 'runs/z13_spatial_v2_20261013'


def main():
    import argparse
    import msvcrt
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe-only', action='store_true')
    args = ap.parse_args()
    lock = (RUN / 'controller.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    jobfolder = RUN / ('controller_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    jobfolder.mkdir()
    flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
    stop = RUN / 'stop.request'
    current = None
    base = {'controller_pid': os.getpid(), 'controller_directory': str(jobfolder), 'authorized_scope': protocol['scope']}

    def status(**kw):
        doc = {**base, **kw, 'updated': datetime.now().astimezone().isoformat()}
        atomic_json(RUN / 'status.json', doc)
        atomic_json(jobfolder / 'status.json', doc)

    def kill_owned():
        if current is not None and current.poll() is None:
            subprocess.run(['taskkill', '/PID', str(current.pid), '/T', '/F'], creationflags=subprocess.CREATE_NO_WINDOW,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            current.wait(timeout=30)

    def run(stage, switches, progress_root):
        nonlocal current
        for attempt in range(1, 4):
            if stop.exists():
                status(state='paused', stage=stage); return 75
            env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
            with (jobfolder / 'log.txt').open('a', encoding='utf-8') as log:
                log.write(f'\n[{stage}] attempt={attempt} {datetime.now().astimezone().isoformat()}\n'); log.flush()
                current = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u',
                    'scripts/z13_spatial_train.py', *switches], cwd=ROOT, env=env, creationflags=flags,
                    stdout=log, stderr=subprocess.STDOUT)
                status(state='running', stage=stage, attempt=attempt, child_pid=current.pid)
                last_progress = time.monotonic(); last_stamp = 0; stalled = False
                while current.poll() is None:
                    time.sleep(5)
                    paths = list(progress_root.rglob('status.json')) if progress_root.exists() else []
                    stamp = max((p.stat().st_mtime for p in paths), default=0)
                    if stamp > last_stamp:
                        last_stamp = stamp; last_progress = time.monotonic()
                    if time.monotonic() - last_progress > protocol['safety']['stall_seconds']:
                        stalled = True
                        log.write(f'\n[WATCHDOG] no progress {protocol["safety"]["stall_seconds"]} s; stopping own child\n'); log.flush()
                        kill_owned(); break
                code = current.returncode
                current = None
            if code == 0:
                return 0
            if code == 75 or stop.exists():
                status(state='paused', stage=stage, exit_code=code); return 75
            if not stalled:
                status(state='failed', stage=stage, exit_code=code, reason='Worker error; no automatic retuning')
                return code or 1
            status(state='retrying' if attempt < 3 else 'failed', stage=stage, attempt=attempt,
                   reason='Owned child stalled; resume only from verified checkpoint')
        return 2

    try:
        technical = RUN / 'technical/report.json'
        if not technical.exists() or not json.loads(technical.read_text(encoding='utf-8')).get('passed'):
            code = run('technical', ['--probe'], RUN / 'technical')
            if code: return code
        if not json.loads(technical.read_text(encoding='utf-8')).get('passed'):
            status(state='failed', stage='technical'); return 2
        if args.probe_only:
            status(state='technical_passed', stage='technical'); return 0
        # Check unchanged code/data before reusing results from an interrupted controller.
        from scripts.z13_spatial_train import identity
        for index, job in enumerate(protocol['jobs']):
            folder = RUN / job['arm'] / job['name']
            ident = identity(protocol, job)
            if (folder / 'identity.json').exists() and json.loads((folder / 'identity.json').read_text(encoding='utf-8')) != ident:
                raise ValueError('Existing job identity changed; automatic reuse forbidden')
            done = folder / 'status.json'
            completed = done.exists() and json.loads(done.read_text(encoding='utf-8')).get('stage') == 'predictions_saved'
            if not completed:
                code = run(f'{job["arm"]}_{job["name"]}', ['--job', str(index)], folder)
                if code: return code
            if job['name'] == 'full':
                code = run(f'{job["arm"]}_quality', ['--quality', job['arm']], RUN / job['arm'] / 'quality')
                if code: return code
        summary = {a: json.loads((RUN / a / 'tau.json').read_text(encoding='utf-8')) for a in protocol['arms']}
        atomic_json(RUN / 'training_result.json', {'completed': datetime.now().astimezone().isoformat(),
                    'arms': summary, 'hota_measured': False, 'heldout_opened': False,
                    'decision': 'Train-only gate complete; no automatic dev/MASA run'})
        status(state='training_completed', stage='B_C_D', tau={a: r['tau'] for a, r in summary.items()})
        return 0
    except Exception as exc:
        status(state='failed', stage='controller', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        kill_owned(); lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
