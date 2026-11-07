"""Независимая сверка границы через разность логарифмов шансов."""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/b1_real_inputs_20261026'
OUT = ROOT / 'runs/b123_20261027/b1_residual_bound'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def logit(p):
    return math.log(p) - math.log1p(-p) if 0 < p < 1 else (-math.inf if p == 0 else math.inf)


def main():
    config, report = read(BASE/'protocol.json'), read(OUT/'report.json')
    with np.load(BASE/'base_inputs.npz') as z:
        uid, names, mask = z['uid'], z['name'], z['mask']
    with np.load(BASE/'base_predictions.npz') as z:
        probability = z['identity_probability']
    with np.load(BASE/'supervision.npz') as z:
        labels = z['ident']
    allowed = {}
    for video in config['fit84'] + config['fit20']:
        allowed.update({r['uid']: r['has_history'] for r in read(BASE/'inputs'/(video+'.json'))['rows']})
    counts, examples = {}, []
    for split in ('fit84', 'fit20'):
        eligible = wrong = impossible = 0
        for i, u in enumerate(uid):
            if names[i] not in config[split] or not allowed[int(u)]:
                continue
            k = [k for k in range(1, mask.shape[1]) if mask[i, k]]
            if any(labels[i, j] < 0 for j in k):
                continue
            positive = [probability[i, j] for j in k if labels[i, j] == 1]
            negative = [probability[i, j] for j in k if labels[i, j] == 0]
            if not positive or not negative:
                continue
            eligible += 1
            t, f = max(positive), max(negative)
            wrong += int(f > t)
            blocked = logit(f) - logit(t) > 2 + 1e-9
            impossible += int(blocked)
            if blocked and split == 'fit20':
                examples.append(dict(uid=int(u), video=str(names[i]), true_p=float(t), false_p=float(f),
                                     logit_margin=logit(f)-logit(t)))
        got = dict(eligible_queries=eligible, baseline_ranking_errors=wrong, impossible_ranking_errors=impossible)
        assert all(v == report['splits'][split][k] for k, v in got.items())
        counts[split] = got
    audit = dict(passed=True, independent_formula='logit(max false p)-logit(max true p)>2',
                 counts=counts, fit20_counterexamples=examples,
                 scope='Current bounded identity ranking only; does not imply failure of action policy, HOTA, or B1-B3 as families.')
    (OUT/'independent_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
