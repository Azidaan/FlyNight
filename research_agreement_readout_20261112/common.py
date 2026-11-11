"""Utilities for the explicitly bounded saved-map agreement experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TRAIN = ROOT / 'research_domain_matched_reader_20261109'
SCALE = ROOT / 'research_signal_scaling_20261109'
NATIVE = ROOT / 'runs/domain_matched_native_20261109'
STABLE = ROOT / 'research_identity_training_stability_20261109'
ARMS = ['C34', 'R134_spatial', 'R134_temporal']
KINDS = ['agreement', 'variance_only']
ID = 'map_value_agreement_20261112'
MSK = timezone(timedelta(hours=3))
STOP = datetime(2026, 11, 12, 7, 40, tzinfo=MSK)
sys.path.insert(0, str(ROOT))

def now():
    return datetime.now(MSK)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cast(x):
    if isinstance(x, dict):
        return {k: cast(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [cast(v) for v in x]
    return x.tolist() if hasattr(x, 'tolist') else x

def save(name, data):
    with (OUT / name).open('x', encoding='utf8') as f:
        json.dump(cast(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def status(stage, **kw):
    assert now() < STOP, 'Authorized work window ended'
    temp = OUT / 'status.pending'
    temp.write_text(json.dumps(dict(stage=stage, updated=now().isoformat(), **kw), ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(OUT / 'status.json')

def guard():
    sys.path.insert(0, str(ROOT / 'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    remaining = (STOP - now()).total_seconds()
    assert remaining > 0
    guard_entrypoint(timeout=remaining)
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)

def key(video, sequence):
    return hashlib.sha256(json.dumps([video, sequence], separators=(',', ':')).encode()).hexdigest()[:32]

def verify():
    p = read(OUT / 'protocol.json')
    for n, h in {**p['sources'], **p['inputs']}.items():
        assert sha(ROOT / n) == h, n
    return p

def checked_edit(path, transform):
    path = ROOT / path
    before = path.read_bytes()
    after = transform(before.decode('utf8')).encode('utf8')
    assert path.read_bytes() == before, f'Concurrent change: {path}'
    path.write_bytes(after)

def journal(text):
    entry = '**' + now().strftime('%d.%m.%Y, %H:%M MSK') + '** — ' + text + '\n\n'
    def update(s):
        anchor = '## Журнал изменений\n'
        assert s.count(anchor) == 1
        return s.replace(anchor, anchor + '\n' + entry, 1)
    checked_edit('dissertation/ЕДИНЫЙ_ПЛАН.md', update)
    with (OUT / 'journal_additions.txt').open('a', encoding='utf8') as f:
        f.write(entry)

def rewrite_json(name, callback):
    def change(s):
        d = json.loads(s)
        callback(d)
        return json.dumps(d, ensure_ascii=False, indent=2) + '\n'
    checked_edit(name, change)

def arrays(snapshot):
    import numpy as np
    values = {}
    for item in snapshot['files']:
        path = ROOT / item['path']
        if item['array'] is None:
            a = np.load(path, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as z:
                a = z[item['array']].copy()
        slot = item['slot']
        if slot in values:
            assert np.array_equal(values[slot], a)
        values[slot] = a
    if 'C34' not in values:
        c = np.empty((721, 34), dtype=np.float32)
        c[:, 4:12] = values['C8']
        c[:, [0, 1, 2, 3] + list(range(12, 34))] = values['C_OTHER']
        values['C34'] = c
    return values

def representations(values, neighbors):
    import numpy as np
    r = values['R1'].astype(np.float64)
    spatial = np.concatenate([r[:, :17], r[neighbors[:, 1:], :17].mean(axis=1)], axis=1)
    result = dict(C34=values['C34'].astype(np.float64), R134_temporal=r, R134_spatial=spatial)
    assert all(a.shape == (721, 34) and np.isfinite(a).all() for a in result.values())
    return result
