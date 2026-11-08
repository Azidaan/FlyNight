import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_identity_assignment_run_20261108.common import read, write, sha, datetime, timedelta, MSK

RUN = ROOT / 'runs/singleton_working_domain_20261108'


def phase():
    p = read(RUN / 'protocol.json')
    for name, digest in p['sources'].items():
        assert sha(ROOT / name) == digest, name
    return p


def status(stage, **values):
    previous = read(RUN / 'status.json') if (RUN / 'status.json').exists() else {}
    previous.update(stage=stage, updated=datetime.now(MSK).isoformat(), **values)
    write(RUN / 'status.json', previous)


def guard(timeout=900):
    sys.path.insert(0, str(ROOT / 'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    if (RUN / 'execution_window.json').exists():
        deadline = datetime.fromisoformat(read(RUN / 'execution_window.json')['deadline'])
        timeout = min(timeout, max(1, int((deadline - datetime.now(MSK)).total_seconds())))
    return guard_entrypoint(timeout=timeout)
