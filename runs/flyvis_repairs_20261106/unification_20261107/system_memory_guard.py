"""Audit-only launcher: sequential workers, system reserve, no allocation cap."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import uuid
import msvcrt
import psutil

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
MARK = '--audit-memory-worker='


def memory_sample():
    v = psutil.virtual_memory()
    return {'available': v.available, 'total': v.total}


def stop_owned(owned, marker):
    """Require both creation time and the unique launch argument before killing."""
    errors = []
    for pid, birth in reversed(list(owned.items())):
        try:
            p = psutil.Process(pid)
            if p.create_time() != birth or marker not in p.cmdline():
                errors.append({'pid': pid, 'reason': 'identity_mismatch'})
                continue
            p.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            errors.append({'pid': pid, 'reason': 'access_denied'})
    return errors


def supervise(command, marker, report_path, *, sample=memory_sample,
              reserve_bytes=None, timeout=600):
    initial = sample()
    reserve = reserve_bytes if reserve_bytes is not None else max(6 * GIB, int(initial['total'] * .20))
    result = dict(command=command, limit_active=False, limit_bytes=None,
                  reserve_bytes=reserve, minimum_available_bytes=initial['available'],
                  reason=None, returncode=None, owned={}, stop_errors=[])
    def save():
        report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if initial['available'] < reserve + 2 * GIB:
        result.update(reason='insufficient_start_reserve', returncode=125)
        save()
        return result
    save()
    child = None
    worker_log = None
    started = time.monotonic()
    try:
        worker_log = (report_path.parent/(report_path.stem + '.worker.log')).open('wb')
        child = subprocess.Popen(command, stdout=worker_log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        owner = psutil.Process(child.pid)
        result['owned'][owner.pid] = owner.create_time()
        save()
        while child.poll() is None:
            try:
                for p in owner.children(recursive=True):
                    if marker in p.cmdline():
                        result['owned'].setdefault(p.pid, p.create_time())
            except psutil.NoSuchProcess:
                pass
            current = sample()
            result['minimum_available_bytes'] = min(result['minimum_available_bytes'], current['available'])
            if current['available'] < reserve:
                result['reason'] = 'system_memory_reserve'
                break
            if time.monotonic() - started > timeout:
                result['reason'] = 'timeout'
                break
            time.sleep(.1)
        if result['reason']:
            result['stop_errors'] = stop_owned(result['owned'], marker)
        try:
            result['returncode'] = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            result.update(reason='worker_did_not_stop', returncode=125)
    except BaseException as exc:
        result.update(reason='supervisor_error', error=repr(exc), returncode=125)
        if child is not None:
            result['stop_errors'] = stop_owned(result['owned'], marker)
        raise
    finally:
        if worker_log is not None:
            worker_log.close()
        result['seconds'] = time.monotonic() - started
        save()
    return result


class Observation:
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def read(self):
        return dict(limit_active=False, limit_bytes=None,
                    available_bytes=psutil.virtual_memory().available,
                    policy='system_reserve_external_supervisor',
                    supervisor_report=str(self.run_dir / 'memory_guard.json'))


def guard_entrypoint(*, timeout=600):
    """Call before importing Torch. Re-enter only below a verified supervisor."""
    if sys.argv[-1].startswith(MARK):
        marker = sys.argv.pop()
        run_dir = HERE / 'uncapped_checks' / marker[len(MARK):]
        state = json.loads((run_dir / 'supervisor.json').read_text(encoding='utf-8'))
        p = psutil.Process(state['pid'])
        assert p.create_time() == state['birth']
        assert any(a.pid == p.pid for a in psutil.Process().parents())
        return Observation(run_dir)
    # A shared OS file lock prevents two revised audit launchers running together.
    with (HERE / 'sequential_memory_guard.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('Another audit verification is running; no second worker was started.')
        try:
            token = uuid.uuid4().hex
            run_dir = HERE / 'uncapped_checks' / token
            run_dir.mkdir(parents=True, exist_ok=False)
            p = psutil.Process()
            (run_dir / 'supervisor.json').write_text(json.dumps({'pid': p.pid, 'birth': p.create_time()}), encoding='utf-8')
            marker = MARK + token
            command = [sys.executable, '-u', str(Path(sys.argv[0]).resolve()), *sys.argv[1:], marker]
            report = supervise(command, marker, run_dir / 'memory_guard.json', timeout=timeout)
            print('SYSTEM_MEMORY_GUARD ' + json.dumps(report), flush=True)
            raise SystemExit(125 if report['reason'] else report['returncode'])
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
