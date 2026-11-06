"""Bounded diagnostic; every target is a direct Popen child owned by this script."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import z13_window_runner as runner

target = Path(__file__).with_name('stop_guard_probe.json')
report = {'command': 'owned python time.sleep(20)', 'taskkill': []}
def save():
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

real_run = subprocess.run
def capture(args, **kwargs):
    kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = real_run(args, **kwargs)
    report['taskkill'].append(dict(args=args, returncode=result.returncode,
        stdout=result.stdout.decode('cp866', errors='replace'), stderr=result.stderr.decode('cp866', errors='replace')))
    save()
    return result

flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
own = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], creationflags=flags)
other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], creationflags=flags)
report.update(owned_pid=own.pid, sibling_pid=other.pid)
save()
try:
    runner.subprocess.run = capture
    runner.stop_owned_child(own)
    try:
        report['owned_exit_after_stop'] = own.wait(timeout=2)
    except subprocess.TimeoutExpired:
        report['owned_exit_after_stop'] = 'still_running'
    report['sibling_alive_after_stop'] = other.poll() is None
finally:
    runner.subprocess.run = real_run
    report['cleanup'] = []
    for process in (own, other):
        row = {'pid': process.pid, 'alive': process.poll() is None}
        if process.poll() is None:
            try:
                process.terminate()
                row['handle_terminate_exit'] = process.wait(timeout=5)
            except Exception as error:
                row['handle_terminate_error'] = repr(error)
                # The only fallback is waiting for this owned bounded sleep.
                row['natural_exit'] = process.wait(timeout=25)
        report['cleanup'].append(row)
    report['all_owned_children_finished'] = all(p.poll() is not None for p in (own, other))
    save()
print(json.dumps(report, ensure_ascii=False, indent=2))
