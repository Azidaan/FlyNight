"""Sequential, resumable inference controller; full budget must be explicitly approved."""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'
FILES = ['tracking/z13_spatial_localizer.py', 'tracking/z13_spatial.py', 'tracking/z13_readout.py',
         'scripts/build_z13_spatial_hota.py', 'scripts/build_fly_signatures.py',
         'scripts/z13_spatial_hota_worker.py', 'scripts/z13_spatial_hota_eval.py',
         'scripts/z13_spatial_hota_runner.py', 'scripts/build_masa_embeddings.py',
         'tracking/masa_r50.py', 'scripts/z13_eval.py', 'scripts/stitch_eval_v2.py',
         'scripts/eval_fly_reid.py', 'scripts/freeze_bench.py', 'tracking/xywh_bytetrack.py']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', action='store_true'); args = ap.parse_args()
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    jobs = [{'id': f'{a}_sig_{v}', 'stage': 'signatures', 'arm': a, 'video': v}
            for a in protocol['arms'] for v in protocol['videos']]
    jobs += [{'id': f'{a}_masa', 'stage': 'masa', 'arm': a} for a in protocol['arms']]
    jobs += [{'id': 'eval', 'stage': 'eval'}]
    if args.plan:
        print(json.dumps({'jobs': len(jobs), 'signature_jobs': 105, 'masa_jobs': 3,
                          'requires': 'technical/report.json passed + full_authorization.json'})); return
    import msvcrt
    authorization = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    assert authorization['approved'] is True and authorization['maximum_seconds'] > 0
    deadline = datetime.fromisoformat(authorization['deadline'])
    assert deadline.tzinfo is not None and datetime.now().astimezone() < deadline, 'Approved window expired'
    assert json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))['passed']
    lock = (RUN / 'controller.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    paths = FILES + ['runs/z13_spatial_hota_v1_20261014/protocol.json']
    paths += [f'runs/z13_spatial_hota_v1_20261014/{a}/model.pt' for a in protocol['arms']]
    paths += ['weights/masa/masa_r50.pth', 'runs/flyvis/results/flow/0000/000/best_chkpt',
              'runs/train_cache/selection.json', 'tracking/retina.py', 'tracking/foveated_eyes.py']
    for video in protocol['videos']:
        paths += [f'runs/train_cache/{video}/detections/raw_video.npz',
                  f'runs/train_cache/{video}/motion/flyvis_tiled.json',
                  f'data/bdd100k/videos/train/{video}.mov']
    def digest(path):
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
        return sha.hexdigest()
    identity = {p: digest(ROOT / p) for p in paths}
    contract = RUN / 'inference_identity.json'
    if contract.exists():
        assert json.loads(contract.read_text(encoding='utf-8')) == identity, 'Code or model changed'
    else:
        atomic_json(contract, identity)
    folder = RUN / ('controller_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
    began = time.monotonic(); child = None
    markers = RUN / 'completed_jobs'; markers.mkdir(exist_ok=True)
    try:
        for index, job in enumerate(jobs):
            marker = markers / (job['id'] + '.json')
            if marker.exists():
                assert json.loads(marker.read_text(encoding='utf-8'))['identity'] == identity
                continue
            if ((RUN / 'stop.request').exists() or time.monotonic() - began >= authorization['maximum_seconds']
                    or datetime.now().astimezone() >= deadline):
                atomic_json(RUN / 'status.json', {'state': 'paused', 'completed_jobs': index, 'total_jobs': len(jobs)}); return
            switches = ['--stage', job['stage']]
            for key in ('arm', 'video'):
                if key in job:
                    switches += ['--' + key, job[key]]
            with (folder / 'log.txt').open('a', encoding='utf-8') as log:
                log.write('\n' + job['id'] + '\n'); log.flush()
                child = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u',
                    'scripts/z13_spatial_hota_worker.py', *switches], cwd=ROOT, creationflags=flags,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                         'OPENBLAS_NUM_THREADS': '2'}, stdout=log, stderr=log)
                last_progress = time.monotonic(); last_stamp = 0
                while child.poll() is None:
                    elapsed = time.monotonic() - began
                    stamp = (folder / 'log.txt').stat().st_mtime
                    if stamp > last_stamp:
                        last_stamp = stamp; last_progress = time.monotonic()
                    atomic_json(RUN / 'status.json', {'state': 'running', 'controller_pid': os.getpid(),
                        'child_pid': child.pid, 'job': job, 'completed_jobs': index, 'total_jobs': len(jobs),
                        'elapsed_seconds': elapsed, 'maximum_seconds': authorization['maximum_seconds'],
                        'log': str(folder / 'log.txt'), 'updated': datetime.now().astimezone().isoformat()})
                    if (elapsed >= authorization['maximum_seconds'] or (RUN / 'stop.request').exists()
                            or datetime.now().astimezone() >= deadline):
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                            creationflags=subprocess.CREATE_NO_WINDOW, stdout=log, stderr=log)
                        child.wait(timeout=30)
                        atomic_json(RUN / 'status.json', {'state': 'paused', 'completed_jobs': index,
                            'total_jobs': len(jobs), 'reason': 'budget or stop request; completed videos preserved'})
                        return
                    if time.monotonic() - last_progress > 900:
                        raise RuntimeError('No log progress for 900 seconds; stopping own worker')
                    time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(f"{job['id']} failed: {child.returncode}; no retuning")
                child = None
            atomic_json(marker, {'identity': identity, 'job': job, 'completed': datetime.now().astimezone().isoformat()})
        atomic_json(RUN / 'status.json', {'state': 'completed', 'completed_jobs': len(jobs),
            'total_jobs': len(jobs), 'report': str(RUN / 'eval/report.json'),
            'updated': datetime.now().astimezone().isoformat()})
    except Exception as exc:
        atomic_json(RUN / 'status.json', {'state': 'failed', 'reason': str(exc), 'updated': datetime.now().astimezone().isoformat()})
        raise
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], creationflags=subprocess.CREATE_NO_WINDOW)
        lock.close()


if __name__ == '__main__':
    main()
