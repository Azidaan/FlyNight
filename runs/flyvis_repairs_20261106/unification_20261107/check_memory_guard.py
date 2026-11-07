"""Check launcher control paths with tiny children; never exhaust real memory."""
from pathlib import Path
import json
import subprocess
import sys
import time
import uuid
import msvcrt
import psutil
from system_memory_guard import GIB, MARK, guard_entrypoint, supervise, stop_owned

HERE = Path(__file__).resolve().parent
if '--probe' in sys.argv:
    job = guard_entrypoint()
    state = job.read()
    assert state['limit_active'] is False and state['limit_bytes'] is None
    (job.run_dir/'probe.json').write_text(json.dumps(state), encoding='utf-8')
    raise SystemExit(0)

out = HERE/('memory_guard_validation_' + uuid.uuid4().hex[:8])
out.mkdir(exist_ok=False)
checks = {}
command = [sys.executable, '-u', str(Path(__file__).resolve()), '--probe']
p = subprocess.run(command, capture_output=True, text=True, timeout=20)
(out/'entrypoint.log').write_text(p.stdout+p.stderr, encoding='utf-8')
assert p.returncode == 0, p.stdout+p.stderr
checks['guarded_entrypoint_without_cap'] = True

with (HERE/'sequential_memory_guard.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        blocked = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert blocked.returncode != 0 and 'no second worker' in blocked.stderr
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
checks['concurrent_audit_launch_is_rejected'] = True

marker = MARK + uuid.uuid4().hex
tiny = [sys.executable, '-c', 'import time; time.sleep(10)', marker]
def high(): return dict(available=20*GIB,total=32*GIB)
def low(): return dict(available=GIB,total=32*GIB)
r = supervise(tiny, marker, out/'refused.json', sample=low, reserve_bytes=6*GIB)
assert r['reason']=='insufficient_start_reserve' and not r['owned']
checks['refuses_start_without_system_reserve'] = True

count = 0
def falling():
    global count
    count += 1
    return high() if count<4 else low()
r = supervise(tiny, marker, out/'stopped.json', sample=falling, reserve_bytes=6*GIB)
assert r['reason']=='system_memory_reserve' and r['returncode'] != 0 and not r['stop_errors']
time.sleep(.2)
for pid,birth in r['owned'].items():
    try: assert psutil.Process(pid).create_time()!=birth
    except psutil.NoSuchProcess: pass
checks['stops_only_launched_worker_when_reserve_falls'] = True

me = psutil.Process()
assert stop_owned({me.pid:me.create_time()+1}, marker)[0]['reason']=='identity_mismatch'
assert stop_owned({me.pid:me.create_time()}, marker)[0]['reason']=='identity_mismatch'
checks['wrong_birth_or_launch_marker_cannot_kill_process'] = True

for name in ['per_file.py','pipeline_smoke.py','run_regression.py','manual_cpu.py']:
    content=(HERE/name).read_text(encoding='utf-8')
    compile(content, name, 'exec')
    assert 'MemoryJob(' not in content and 'guard_entrypoint()' in content
checks['four_active_launchers_have_no_explicit_job_cap'] = True
result=dict(passed=True,checks=checks,torch_imported='torch' in sys.modules,
            real_memory_exhaustion_attempted=False,available_bytes=psutil.virtual_memory().available)
assert not result['torch_imported']
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
print('REPORT '+str(out/'result.json'))
