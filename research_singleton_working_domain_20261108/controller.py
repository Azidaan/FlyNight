import subprocess
import time
from research_singleton_working_domain_20261108.common import *


def main():
    import psutil
    while not (RUN / 'armed.json').exists():
        assert datetime.now(MSK) < datetime.fromisoformat(read(RUN/'execution_window.json')['soft_stop'])
        time.sleep(.1)
    phase()
    status('starting_cpu_worker')
    with (RUN/'worker.log').open('x', encoding='utf-8') as out, (RUN/'worker.err').open('x', encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable, '-u', 'research_singleton_working_domain_20261108/worker.py'],
            cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'worker.started.json', dict(pid=child.pid, created=psutil.Process(child.pid).create_time()), new=True)
        code = child.wait()
    lines = [s for s in (RUN/'worker.log').read_text('utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')]
    assert lines, (code, 'No supervisor completion record')
    guardian = __import__('json').loads(lines[-1].split(' ', 1)[1])
    write(RUN/'worker.ended.json', dict(returncode=code, guardian=guardian), new=True)
    assert code == 0 and guardian['returncode'] == 0 and guardian['reason'] is None
    for pid, birth in guardian['owned'].items():
        try:
            process = psutil.Process(int(pid))
            if abs(process.create_time()-birth) < .01:
                process.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
    phase()
    assert read(RUN/'report.json')['complete'] and read(RUN/'independent_audit.json')['passed']
    write(RUN/'queue_result.json', dict(passed=True, data_only=True,
        report_sha256=sha(RUN/'report.json'), independent_audit_sha256=sha(RUN/'independent_audit.json'),
        no_automatic_followup=True), new=True)


if __name__ == '__main__':
    main()
