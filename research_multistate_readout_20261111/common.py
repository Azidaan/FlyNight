"""Frozen-cohort helpers. No old executor is imported or resumed."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'research_domain_matched_reader_20261109'
MSK = timezone(timedelta(hours=3))
STOP = datetime(2026, 11, 12, 7, 40, tzinfo=MSK)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def cast(x):
    if isinstance(x, dict): return {k: cast(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [cast(v) for v in x]
    return x.tolist() if hasattr(x, 'tolist') else x


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(cast(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def key(video, sequence):
    return hashlib.sha256(json.dumps([video, sequence], separators=(',', ':')).encode()).hexdigest()[:32]


def status(stage, **kwargs):
    path = OUT / 'status.json'
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(cast(dict(stage=stage, updated=datetime.now(MSK).isoformat(), **kwargs)), ensure_ascii=False), encoding='utf8')
    tmp.replace(path)


def start(cpu=True):
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    if cpu: os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    remaining = (STOP - datetime.now(MSK)).total_seconds()
    assert remaining > 0, 'Authorized deadline elapsed'
    guard_entrypoint(timeout=min(1800, remaining))
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)


def verify():
    for name, digest in read(OUT / 'protocol.json')['sources'].items():
        assert sha(ROOT / name) == digest, name


def arrays(k):
    import numpy as np
    phase = read(OLD / 'execution_protocol.json')
    values = {}
    for item in phase['inventory_requests'].get(k, {}).get('ready_files', []):
        with np.load(ROOT / item['path'], allow_pickle=False) as z: a = z[item['array']].copy()
        slot = item['slot']
        if slot in values: np.testing.assert_array_equal(values[slot], a)
        values[slot] = a
    path = OLD / 'maps' / (k + '.npz')
    if path.exists():
        assert read(path.with_suffix('.json'))['sha256'] == sha(path)
        with np.load(path, allow_pickle=False) as z:
            for slot in z.files:
                assert slot not in values
                values[slot] = z[slot].copy()
    if 'C34' not in values and 'C8' in values and 'C_OTHER' in values:
        c = np.empty((721, 34), np.float32)
        c[:, 4:12] = values['C8']
        c[:, [0, 1, 2, 3] + list(range(12, 34))] = values['C_OTHER']
        values['C34'] = c
    return values


def correlations(a, b):
    """Same fixed 34-channel meaning; constant candidate contributes zero."""
    import numpy as np
    from tracking.channel_balanced_correspondence import _center_pair
    x, y, an, bn, active_a, active_b = _center_pair(a, b)
    if not active_a.any() or not active_b.any(): return None
    rho = np.divide((x*y).sum(0), an*bn, out=np.zeros_like(an), where=an*bn > 0)
    return rho * a.shape[1] / active_a.sum()
