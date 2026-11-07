"""Возобновить очередь после подтверждённого позднего выхода зависшего дерева.

Исходный отрицательный controller_result не переписывается. Следующий CUDA
допускается только после отдельной фактической проверки всех PID+birth.
"""
from pathlib import Path
import sys, os, time, inspect, subprocess
import psutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import recovery_research_service_20261027 as service
RUN = service.RUN


def verify_late_end(folder, result, wait_seconds=0):
    end = time.monotonic() + wait_seconds
    while True:
        live = [pid for pid, birth in result['owned'].items() if service.matching(pid, birth)]
        if not live or time.monotonic() >= end: break
        time.sleep(.2)
    assert not live, ('Owned process still alive; queue must remain stopped', live)
    proof = dict(passed=True, actual_all_owned_ended=True, owned=result['owned'],
        original_controller_sha256=service.sha(folder / 'controller_result.json'),
        original_all_owned_ended=result['all_owned_ended'], original_exit_code=result['exit_code'],
        original_failure_preserved=True, updated=service.now())
    service.write(folder / 'late_owned_end_verified.json', proof)
    return proof


def prepare():
    original = service.read(RUN / 'queue.launch.json')
    assert not service.matching(original['pid'], original['created'])
    assert not (RUN / 'queue_recovery1/launch.json').exists()
    folder = RUN / 'b1_tracker/continuation1'
    result = service.read(folder / 'controller_result.json')
    assert result['exit_code'] != 0 and not result['all_owned_ended']
    proof = verify_late_end(folder, result)
    job_path = RUN / 'ready_jobs/330_b1_author_continuation.json'
    target = RUN / 'jobs_done' / job_path.name; assert not target.exists()
    service.write(target, dict(job=service.read(job_path), result=result, finished=service.now(),
        outcome='failed_preserved', late_actual_end=proof, recovery_source=Path(__file__).relative_to(ROOT).as_posix()))
    service.write(RUN / 'queue_recovery1/protocol.json', dict(created=service.now(),
        original_queue_dead=True, original_queue_launch=original, original_failure_log='queue.stderr.log',
        adopted_failed_job=job_path.name, late_process_proof='b1_tracker/continuation1/late_owned_end_verified.json',
        original_controller_unchanged=True, source_sha256=service.sha(__file__),
        deadline_source='authorization.json', no_duplicate_queue=True))


def queue():
    protocol = service.read(RUN / 'queue_recovery1/protocol.json')
    assert service.sha(__file__) == protocol['source_sha256']
    assert not service.matching(protocol['original_queue_launch']['pid'], protocol['original_queue_launch']['created'])
    code = inspect.getsource(service.queue)
    old = "        assert result['all_owned_ended']"
    assert code.count(old) == 1
    code = code.replace(old, "        if not result['all_owned_ended']:\n            verify_late_end(folder,result,20)")
    ns = {**vars(service), 'verify_late_end': verify_late_end}
    exec(compile(code, str(Path(__file__)), 'exec'), ns)
    ns['queue']()


def launch():
    prepare()
    folder = RUN / 'queue_recovery1'
    with (folder / 'stdout.log').open('w', encoding='utf-8') as out, (folder / 'stderr.log').open('w', encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--queue'], cwd=ROOT,
            stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    p = psutil.Process(child.pid); p.nice(psutil.IDLE_PRIORITY_CLASS)
    service.write(folder / 'launch.json', dict(pid=p.pid, created=p.create_time(), source_sha256=service.sha(__file__), launched=service.now()))
    service.register(p, folder)
    service.write(RUN / 'active_queue.json', dict(source='queue_recovery1/launch.json', updated=service.now()))
    assert service.matching(p.pid, p.create_time())
    print('Recovered queue launched', p.pid, p.create_time())


if __name__ == '__main__':
    queue() if '--queue' in sys.argv else launch()
