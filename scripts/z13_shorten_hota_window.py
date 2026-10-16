"""Preserve the old authorization and shorten only the own HOTA time window."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json

RUN = ROOT / 'runs/z13_triton_hota_v1_20261014/full'


def main():
    path = RUN / 'full_authorization.json'
    old = json.loads(path.read_text(encoding='utf-8'))
    deadline = '2026-10-15T07:40:00+03:00'
    assert datetime.now().astimezone() < datetime.fromisoformat(deadline)
    assert datetime.fromisoformat(deadline) < datetime.fromisoformat(old['deadline'])
    guard = RUN / 'deadline_guard_20261015_002024/started.json'
    info = json.loads(guard.read_text(encoding='utf-8'))
    processes = WindowsProcesses()
    creation = processes.creation(info['pid'])
    assert creation is not None
    backup = RUN / 'authorization_before_0740.json'
    assert not backup.exists()
    atomic_json(backup, old)
    updated = {**old, 'deadline': deadline,
               'deadline_revision': {'at': datetime.now().astimezone().isoformat(),
                                    'human_message': 'Смотри. Цель закончить прогон в 7:40',
                                    'previous_deadline': old['deadline']}}
    atomic_json(path, updated)
    atomic_json(RUN / 'deadline_revision_0740.json', {
        'new_deadline': deadline, 'old_guard_pid': info['pid'],
        'old_guard_creation': creation, 'original_controller_reads_old_deadline': True,
        'enforcement': 'New independent guard requests stop 15 seconds before new deadline'})
    print(json.dumps(updated, ensure_ascii=False))


if __name__ == '__main__':
    main()
