"""One-video MASA cost probe, with an independent 180-second owned-worker limit."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def worker():
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import atomic_json
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.set_num_threads(2)
        from scripts.build_masa_embeddings import main as build
        sys.argv = ['masa', '--ids-json', str(RUN / 'probe_video.json'),
                    '--sig-root', str(RUN / 'technical/active_D'),
                    '--videos-dir', str(ROOT / 'data/bdd100k/videos/train'),
                    '--out', str(RUN / 'technical/masa_price/embeddings')]
        began = time.monotonic()
        build()
        atomic_json(RUN / 'technical/masa_price/report.json', {'created': datetime.now().astimezone().isoformat(),
                    'seconds_including_model_load': time.monotonic() - began, 'videos': 1,
                    'scope': 'MASA inference price, no tracking evaluation or HOTA', 'passed': True})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--worker', action='store_true'); args = ap.parse_args()
    if args.worker:
        worker(); return
    from tracking.z13_checkpoint import atomic_json
    folder = RUN / 'technical/masa_price'; folder.mkdir(exist_ok=True)
    assert json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))['passed']
    flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
    began = time.monotonic()
    with (folder / 'log.txt').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u',
                    'scripts/z13_spatial_hota_masa_price.py', '--worker'], cwd=ROOT,
                    creationflags=flags, stdout=log, stderr=log,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        try:
            atomic_json(folder / 'status.json', {'state': 'running', 'child_pid': child.pid,
                        'controller_pid': os.getpid(), 'limit_seconds': 180})
            child.wait(timeout=180)
            atomic_json(folder / 'status.json', {'state': 'completed' if child.returncode == 0 else 'failed',
                        'exit_code': child.returncode, 'seconds': time.monotonic() - began})
        except subprocess.TimeoutExpired:
            atomic_json(folder / 'status.json', {'state': 'stopped_at_limit', 'seconds': time.monotonic() - began})
        finally:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], creationflags=flags,
                               stdout=log, stderr=log)
                child.wait(timeout=30)


if __name__ == '__main__':
    main()
