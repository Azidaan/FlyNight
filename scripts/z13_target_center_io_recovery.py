"""Bounded status-file sharing recovery; preserves the frozen scientific runner."""
import argparse
import hashlib
import json
import os
import runpy
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_target_center_20261018'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(51):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 50:
                raise
            time.sleep(.1)


def worker(budget):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
    os.environ['TIMEZONE'] = 'Europe/Moscow'
    import torch
    torch.cuda.is_available = lambda: False
    torch.set_default_device('cpu')
    assert torch.empty(1).device.type == 'cpu' and not torch.cuda.is_initialized()
    module = runpy.run_path(str(ROOT / 'scripts/z13_target_center_probe.py'), run_name='io_recovery')
    module['main'].__globals__['write'] = write
    sys.argv = ['z13_target_center_probe.py', '--budget', str(budget)]
    try:
        module['main']()
    except Exception as error:
        write(RUN/'status.json', dict(state='paused' if isinstance(error, TimeoutError) else 'failed',
                                      stage=str(error), done=0, total=0))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        from unittest.mock import patch
        target = RUN/'io_recovery_write_test.json'
        original = Path.replace
        attempts = []
        def temporary_conflict(path, destination):
            attempts.append(1)
            if len(attempts) < 4:
                raise PermissionError('simulated Windows sharing conflict')
            return original(path, destination)
        with patch.object(Path, 'replace', temporary_conflict):
            write(target, dict(ok=True, simulated_failures=3))
        assert json.loads(target.read_text(encoding='utf-8'))['ok'] and len(attempts)==4
        print('PASS: three transient sharing failures, then successful atomic replacement')
        return
    if args.worker is not None:
        return worker(args.worker)
    import psutil
    active_names = {'z13_target_center_probe.py', 'z13_target_center_cpu_recovery.py', 'z13_target_center_recovery_controller.py', Path(__file__).name}
    ancestors = {p.pid for p in psutil.Process().parents()} | {os.getpid()}
    for p in psutil.process_iter(['cmdline']):
        if p.pid not in ancestors and any(Path(x).name in active_names for x in (p.info['cmdline'] or [])):
            raise RuntimeError(f'Existing run process {p.pid}')
    result = RUN/'io_recovery_result.json'
    if result.exists() or (RUN/'io_recovery_controller.json').exists():
        raise RuntimeError('Recovery already attempted; inspect before another launch')
    launch = json.loads((RUN/'full_launch.json').read_text(encoding='utf-8-sig'))
    soft = datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', launch['soft_deadline'])).timestamp()
    hard = datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', launch['hard_deadline'])).timestamp()
    budget = int(soft-time.time())
    if budget <= 0:
        raise RuntimeError('Original authorization window ended')
    proto = json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in proto['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    saved = {}
    for arm in ('D','C'):
        assert (RUN/arm/'done.json').exists()
        rep = json.loads((RUN/arm/'reproduction.json').read_text())
        assert rep['lambda_matches'] and rep['max_center_px']<.5 and rep['max_score_difference']<.001
        for file in (RUN/arm).glob('*.json'):
            saved[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    snapshot = RUN/'status_before_io_recovery.json'
    with snapshot.open('xb') as out:
        out.write((RUN/'status.json').read_bytes())
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(budget)]
    write(RUN/'status.json',dict(state='starting',stage='Resume R1; D and C preserved',done=0,total=0))
    with (RUN/'io_recovery.stdout.log').open('w',encoding='utf-8') as out, (RUN/'io_recovery.stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'io_recovery_controller.json',dict(pid=child.pid,command=command,
              started_at=datetime.now(timezone.utc).isoformat(),remaining_budget_seconds=budget,
              soft_deadline=launch['soft_deadline'],hard_deadline=launch['hard_deadline'],
              preserved_sha256=saved,service_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              change='Only atomic JSON-write retry; CPU forced before import. Frozen code unchanged; skip completed D/C.'))
        timeout=False
        try:
            code=child.wait(timeout=max(1,hard-time.time()))
        except subprocess.TimeoutExpired:
            timeout=True
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
            code=child.wait(timeout=20)
        unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in saved.items())
        write(result,dict(exit_code=code,hard_timeout=timeout,worker_ended=True,preserved_D_C_unchanged=unchanged))
        if timeout:
            write(RUN/'status.json',dict(state='paused',stage='Original deadline reached; own worker ended',done=2,total=3))


if __name__=='__main__':
    main()
