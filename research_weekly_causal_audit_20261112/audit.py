"""Read saved evidence; no inference, fitting, tracker replay or threshold search."""
from pathlib import Path
import json
import hashlib
import math
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCES = {}


def read(name):
    p = ROOT / name
    raw = p.read_bytes()
    SOURCES[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(name, value):
    p = OUT / name
    assert not p.exists(), p
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    catalog = read('experiments/project_history.json')
    cards = [c for c in catalog['experiments'] if any(
        '2026-11-05' <= str(c.get(k, ''))[:10] <= '2026-11-12'
        for k in ['started', 'done'])]
    assert len(cards) == 97
    save('inventory.json', dict(period_msk=['2026-11-05', '2026-11-12'],
        selection='started OR done in inclusive date range, before this review card',
        count=len(cards), cards=[{k:c[k] for k in ['id','title','state','started','done','evidence'] if k in c} for c in cards],
        scope='Catalog coverage, not a claim that every historical source was independently reproduced.'))
    feat = read('research_agreement_readout_20261112/features.json')
    pred = read('research_agreement_readout_20261112/predictions.json')
    fit = next(x for x in pred['fits'] if x['video'] == 'FINAL' and x['model'] == 'C34__agreement')
    theta = np.asarray(fit['fit']['theta'], dtype=float)
    norm = fit['normalization']
    rows = []
    for f, p in zip(feat['screen'], pred['screen']):
        assert (f['video'], f['frame'], f['old_id']) == (p['video'], p['frame'], p['old_id'])
        g = (np.asarray(f['geometry']) - norm['mean']) / norm['scale']
        s = (np.asarray(f['variants']['C34']['agreement']) - fit['signal_mean']) / fit['signal_scale']
        contributions = np.r_[g, s] * theta[:-1]
        value = math.fsum([*map(float, contributions), float(theta[-1])])
        expected = p['scores']['C34__agreement']
        assert abs(value - expected) < 1e-12
        rows.append(dict(video=p['video'],frame=p['frame'],old_id=p['old_id'],label=p['label'],
                         geometry=float(math.fsum(contributions[:7])),
                         centered_visual=float(math.fsum(contributions[7:])),
                         intercept=float(theta[-1]),score=value,stored_score=expected,
                         accepted=value>0, per_feature_contributions=contributions.tolist()))
    fate = read('runs/domain_matched_native_20261109/id_fate.json')
    selected = [r for r in fate['records'] if r['arm']=='C34' and r['scored'] and r['kind'] in ['own','foreign']]
    assert len(selected) == 5
    old = read('research_signal_scaling_20261109/deployment_screen.json')
    for row in rows:
        a = next(r for r in old['records'] if r['old_id']==row['old_id'])
        row['old_common_score'] = a['scores']['C34']
        assert (row['score']>0) == (row['old_common_score']>0)
    save('decision_decomposition.json',dict(passed=True, model='C34__agreement, FINAL',rows=rows,
        formula='sum(theta_geometry * standardized_geometry) + sum(theta_signal * centered_scaled_agreement) + intercept',
        reference='Coordinates centered exactly as stored in the final fitted model; contributions are coordinate-dependent arithmetic, not causal attribution.',
        no_new_fits=True,no_new_maps=True,no_tracker_calls=True,
        max_reconstruction_error=max(abs(r['score']-r['stored_score']) for r in rows)))
    save('native_five_cases.json',dict(source='runs/domain_matched_native_20261109/id_fate.json',
        rows=selected, summary=fate['summary']['C34'],
        scope='Historical full tracker, different readout from agreement; no tracker rerun in this review.'))
    for name, h in SOURCES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h, name
    save('receipt.json',dict(passed=True,created_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        sources=SOURCES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calculations='Arithmetic on stored scalars only; inventory and hashes. No new scientific trial.'))
    print(json.dumps(dict(cards=len(cards),rows=[{k:r[k] for k in ['old_id','label','geometry','centered_visual','intercept','score']} for r in rows]),ensure_ascii=False))


if __name__ == '__main__':
    main()
