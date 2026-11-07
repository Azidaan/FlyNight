"""Только проверка сохранности и существующих процессов; расчёты не запускает."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b123_20261027'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def matching(pid, birth):
    try:
        process = psutil.Process(int(pid))
        if process.is_running() and abs(process.create_time() - float(birth)) < .01:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def main():
    previous = read(RUN / 'end_of_measurements.json')
    status = read(RUN / 'deadline_status.json')
    launch = read(RUN / 'deadline_guard_recovery1/launch.json')
    guard = matching(launch['pid'], launch['created'])
    live = []
    for pid, birth in status['owned'].items():
        process = matching(pid, birth)
        if process:
            live.append(dict(pid=int(pid), created=birth, cmdline=process.cmdline()))
    queue_tree = set()
    for entry in previous['queue']:
        queue = matching(entry['pid'], entry['created'])
        if queue:
            queue_tree.update((p.pid, p.create_time()) for p in [queue] + queue.children(recursive=True))
    pending = [p.name for p in (RUN / 'ready_jobs').glob('*.json')
               if not (RUN / 'jobs_done' / p.name).exists()]
    sha = hashlib.sha256((RUN / 'NIGHT_RESULTS.md').read_bytes()).hexdigest()
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(status['updated'])).total_seconds()
    proof = dict(checked=datetime.now(timezone.utc).isoformat(),
                 existing_owned_alive=live, pending_ready_jobs=pending,
                 no_scientific_processes=all((p['pid'], float(p['created'])) in queue_tree for p in live),
                 guard_identity_verified=guard is not None, guard_status_age_seconds=age,
                 guard_source_unchanged=hashlib.sha256((ROOT / 'scripts/b123_deadline_guard_recovery1_20261027.py').read_bytes()).hexdigest() == launch['source_sha256'],
                 night_summary_sha256=sha,
                 night_summary_unchanged=sha == previous['night_summary_sha256'],
                 stop_at=status['stop_at'], new_calculations_started=False)
    proof['passed'] = (proof['no_scientific_processes'] and not pending and guard is not None
                       and 0 <= age < 15 and proof['guard_source_unchanged'] and proof['night_summary_unchanged'])
    dst = RUN / 'final_stop/pre_stop_check_recovery1.json'
    assert not dst.exists(), 'Не перезаписывать прежнюю проверку'
    dst.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False))
    assert proof['passed']


if __name__ == '__main__':
    main()
