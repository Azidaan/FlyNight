"""Account for failed-attempt runtime and idle pause within the 20-minute GPU cap."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN = ROOT / 'runs/z13_cache_speed_v1_20261014'


def main():
    first = next(json.loads(p.read_text(encoding='utf-8')) for p in RUN.glob('deadline_guard_*/result.json')
                 if json.loads(p.read_text(encoding='utf-8'))['state'] == 'controller_failed')
    first_time = first['known_owned_processes']['24184']
    first_start = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=first_time // 10)
    first_end = datetime.fromisoformat(first['ended'])
    used = (first_end - first_start).total_seconds()
    launch = json.loads((RUN / 'launch_process.json').read_text(encoding='utf-8'))
    second_start = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(launch['launcher_start_filetime']) // 10)
    hard = datetime.fromisoformat(json.loads((ROOT / 'runs/z13_triton_hota_v1_20261014/full/full_authorization.json').read_text(encoding='utf-8'))['deadline'])
    deadline = min(second_start + timedelta(seconds=1200 - used), hard).astimezone()
    old_guard = 16504
    creation = WindowsProcesses().creation(old_guard)
    assert creation is not None
    record = {'first_start': first_start.isoformat(), 'first_end': first_end.isoformat(),
              'first_attempt_wall_seconds': used, 'second_start': second_start.isoformat(),
              'maximum_total_active_seconds': 1200, 'remaining_for_second_attempt': 1200 - used,
              'deadline': deadline.isoformat(), 'old_guard_pid': old_guard, 'old_guard_creation': creation,
              'reason': 'Exclude preparation and the verified idle pause; include full process lifetime of both attempts conservatively. HOTA deadline unchanged.'}
    for name in ('full_authorization.json', 'protocol.json'):
        path = RUN / name
        original = json.loads(path.read_text(encoding='utf-8'))
        atomic_json(RUN / (name + '.before_active_accounting.json'), original)
        original['deadline'] = deadline.isoformat()
        original['maximum_total_active_seconds'] = 1200
        original['budget_accounting_source'] = 'runs/z13_cache_speed_v1_20261014/active_budget.json'
        atomic_json(path, original)
    atomic_json(RUN / 'active_budget.json', record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
