"""One bounded owner: wait for successfulqueue4, then run parentqueue5 once."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import sys, time, subprocess, json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import psutil
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():
    return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
spec = read(OUT / 'queue5_wait_spec.json')
for name, digest in spec['inputs'].items():
    assert sha(ROOT / name) == digest, name
assert sha(__file__) == spec['waiter_sha256']
started = time.monotonic()
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
state = dict(started=now(), owner=dict(pid=os.getpid(), birth=psutil.Process().create_time()), stage='waiting_for_queue4', launch_count=0, spec_sha256=sha(OUT / 'queue5_wait_spec.json'))
with (OUT / 'assignment_loss_owner.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(state, ensure_ascii=False, indent=2).rstrip() + '\n')
def persist():
    pending = OUT / 'queue5_waiter_status.pending'
    pending.write_bytes((json.dumps(state, ensure_ascii=False, indent=2).rstrip() + '\n').encode('utf8'))
    pending.replace(OUT / 'queue5_waiter_status.json')
predecessor = ROOT / spec['predecessor_execution']
queue = ROOT / spec['queue']
execution = queue.with_name(queue.stem + '_execution.json')
persist()
while time.monotonic() - started < spec['maximum_seconds'] - 150:
    if execution.exists():
        state.update(stage='queue5_already_exists_not_launched', complete=True, ended=now())
        persist()
        raise SystemExit(0)
    if predecessor.exists():
        try:
            previous = read(predecessor)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(2)
            continue
        if previous.get('complete'):
            if not previous.get('passed'):
                state.update(stage='predecessor_failed_no_launch', complete=True, ended=now(), predecessor_sha256=sha(predecessor))
                persist()
                raise SystemExit(2)
            for name, digest in spec['inputs'].items():
                assert sha(ROOT / name) == digest, name
            prior_spec_path = predecessor.with_name('queue4.json')
            prior_spec = read(prior_spec_path)
            assert len(previous['jobs']) == len(prior_spec['jobs'])
            assert all(job['result']['returncode'] == 0 and job['result']['reason'] is None for job in previous['jobs'])
            assert [job['entry'] for job in previous['jobs']] == [job['entry'] for job in prior_spec['jobs']]
            assert prior_spec['runner_sha256'] == sha(ROOT / spec['runner'])
            state.update(stage='launching_queue5', predecessor_sha256=sha(predecessor), predecessor_spec_sha256=sha(prior_spec_path), launch_count=1)
            persist()
            child = subprocess.Popen([sys.executable, '-u', str(ROOT / spec['runner']), str(queue)], cwd=ROOT)
            state['child'] = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time())
            state['stage'] = 'queue5_running'
            persist()
            code = child.wait(timeout=145)
            completed = read(execution)
            state.update(stage='complete' if code == 0 and completed.get('passed') else 'queue5_failed', complete=True, ended=now(), returncode=code,
                         queue5_execution_sha256=sha(execution), queue5_passed=completed.get('passed'))
            persist()
            raise SystemExit(code)
    time.sleep(3)
state.update(stage='bounded_wait_expired_no_launch', complete=True, ended=now())
persist()
raise SystemExit(3)
