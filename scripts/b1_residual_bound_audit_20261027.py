"""Проверка границы текущей Б1 по сохранённым данным; без Torch и обучения."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/b1_real_inputs_20261026'
OUT = ROOT / 'runs/b123_20261027/b1_residual_bound'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare():
    OUT.mkdir(exist_ok=False)
    p = read(BASE / 'protocol.json')
    paths = [Path(__file__), ROOT / 'tracking/flyvis_temporal_residual.py',
             ROOT / 'tracking/b1_real_inputs.py']
    paths += [BASE / n for n in ('base_inputs.npz', 'base_predictions.npz', 'supervision.npz', 'protocol.json')]
    paths += [BASE / 'inputs' / (v + '.json') for v in p['fit84'] + p['fit20']]
    write(OUT / 'protocol.json', dict(created=datetime.now(timezone.utc).isoformat(), budget_seconds=60,
        question='Are some known identity ranking errors impossible to reverse with current bounded residual?',
        cohort='Observable history; at least two real regions; every real identity label known; both true and false regions present. NULL excluded. Splits reported separately.',
        criterion='For each false region minimum achievable p exceeds maximum achievable p of every true region, using independent deltas in [-1,1]. Strict inequality with 1e-12 numerical tolerance.',
        bound='p_delta = p*exp(delta)/(1-p+p*exp(delta)); all independent corrections allowed as an optimistic superset of actual head outputs. Endpoints 0 and 1 remain fixed.',
        limitation='Only current fixed candidate identity ranking, not whole policy or HOTA. Does not reject other B1 implementations, B2, B3, or refusal to intervene. Reused development data.',
        sha256={str(q.relative_to(ROOT)).replace('\\', '/'): sha(q) for q in paths}))


def run():
    start = time.monotonic()
    protocol = read(OUT / 'protocol.json')
    for name, digest in protocol['sha256'].items():
        assert sha(ROOT / name) == digest, name
    p = read(BASE / 'protocol.json')
    with np.load(BASE / 'base_inputs.npz') as z:
        uid, name, mask = z['uid'], z['name'], z['mask'].copy()
    with np.load(BASE / 'base_predictions.npz') as z:
        assert np.array_equal(z['uid'], uid)
        probability = z['identity_probability']
    with np.load(BASE / 'supervision.npz') as z:
        assert np.array_equal(z['uid'], uid)
        ident = z['ident']
    history = {}
    for v in p['fit84'] + p['fit20']:
        history.update({r['uid']: bool(r['has_history']) for r in read(BASE / 'inputs' / (v + '.json'))['rows']})
    mask[:, 0] = False
    has_history = np.array([history[int(u)] for u in uid])
    assert np.isfinite(probability[mask]).all()
    assert ((probability[mask] >= 0) & (probability[mask] <= 1)).all()
    safe = np.where(mask, probability, .5)
    lower = safe * np.exp(-1) / (1 - safe + safe * np.exp(-1))
    upper = safe * np.exp(1) / (1 - safe + safe * np.exp(1))
    true = mask & (ident == 1)
    false = mask & (ident == 0)
    complete = (~mask | (ident >= 0)).all(1)
    eligible = has_history & complete & true.any(1) & false.any(1)
    true_upper = np.where(true, upper, -np.inf).max(1)
    false_lower = np.where(false, lower, -np.inf).max(1)
    impossible = eligible & (false_lower > true_upper + 1e-12)
    wrong = eligible & (np.where(false, safe, -np.inf).max(1) > np.where(true, safe, -np.inf).max(1))
    assert not (impossible & ~wrong).any()
    result = dict(protocol_sha256=sha(OUT / 'protocol.json'), splits={}, examples=[],
                  statement='Conditional impossibility for bounded current ranking only; no HOTA measurement.')
    for split in ('fit84', 'fit20'):
        selected = np.isin(name, p[split])
        n = int((wrong & selected).sum())
        k = int((impossible & selected).sum())
        result['splits'][split] = dict(queries=int(selected.sum()), eligible_queries=int((eligible & selected).sum()),
            baseline_ranking_errors=n, impossible_ranking_errors=k,
            impossible_share_of_errors=k/n if n else None,
            incomplete_labels_with_true_and_false=int((selected & has_history & ~complete & true.any(1) & false.any(1)).sum()))
    for i in np.flatnonzero(impossible)[:10]:
        result['examples'].append(dict(uid=int(uid[i]), video=str(name[i]),
            true_p=probability[i, true[i]].tolist(), false_p=probability[i, false[i]].tolist(),
            best_true_upper=float(true_upper[i]), best_false_lower=float(false_lower[i])))
    result['elapsed_seconds'] = time.monotonic() - start
    assert result['elapsed_seconds'] < protocol['budget_seconds']
    write(OUT / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
