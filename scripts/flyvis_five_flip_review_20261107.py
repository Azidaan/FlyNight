"""Post-hoc accounting of five published decisions, without executing FlyVis."""
import csv
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

BASE = Path('runs/flyvis_exclusive_t4t5_20261107')
OUT = BASE / 'five_flips'
MAP = Path('runs/z13_spatial_set_readout_20261025/recovery2/features/C/train26')
SEED = Path('runs/z13_region_assignment_20261025/recovery2/features/C/train26')
LABEL = Path('runs/z13_temporal_pair_readiness_20261025')
CH = [f'T{n}{letter}' for n in (4, 5) for letter in 'abcd']


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def load(p):
    return json.loads(p.read_bytes())


def review():
    assert not (OUT / 'decomposition.json').exists()
    fp = load(BASE / 'inputs_used.json')
    measurement = load(BASE / 'measurement.json')
    bound = {}

    def bind(path, expected):
        h = sha(path)
        assert h == expected, path
        bound[path.as_posix()] = h

    bind(BASE / 'inputs_used.json', measurement['inputs_sha256'])
    bind(BASE / 'choices.csv.gz', measurement['rows_sha256'])
    bind(BASE / 'protocol.json', measurement['protocol_sha256'])
    bind(BASE / 'measurement.json', load(BASE / 'report.json')['measurement_sha256'])
    assert CH == measurement['channels']
    with gzip.open(BASE / 'choices.csv.gz', 'rt', encoding='ascii') as f:
        primary = [z for z in csv.DictReader(f) if z['split'] == 'evaluate26'
                   and z['exactly2'] == '1' and z['label_reason'] == 'ok' and z['exclusive_choice']]
    cases = {int(z['uid']): z for z in primary if z['whole_choice'] != z['exclusive_choice']}
    assert len(primary) == 172 and len(cases) == len({z['video'] for z in cases.values()}) == 5

    def locate(base):
        result = {}
        for path in sorted(base.glob('chunk_*.npz')):
            with np.load(path, allow_pickle=False) as z:
                for i, u in enumerate(z['uid']):
                    uid = int(u)
                    if uid in cases:
                        assert uid not in result
                        result[uid] = (path, i)
        assert result.keys() == cases.keys()
        for path in set(p for p, _ in result.values()):
            bind(path, fp[path.as_posix()])
        return result

    maps, seeds = locate(MAP), locate(SEED)
    records = []
    worst = 0.
    for uid, z in cases.items():
        v = z['video']
        side = int(z['target_side'])
        assert side == int(z['whole_choice']) == -int(z['exclusive_choice']) and abs(side) == 1
        paths = [LABEL / name / (v + '.json') for name in ['inputs', 'supervision']]
        for path in paths:
            bind(path, fp[path.as_posix()])
        meta, truth = (next(row for row in load(path)['rows'] if row['uid'] == uid) for path in paths)
        assert meta['current_choices'] == [int(z['j']), int(z['k'])]
        good = 0 if side == 1 else 1
        assert truth['flags']['known'] and all(truth['current_identity']) and truth['target_identity']
        assert [x == truth['target_identity'] for x in truth['current_identity']] == [i == good for i in (0, 1)]
        p, i = maps[uid]
        with np.load(p, allow_pickle=False) as data:
            assert int(data['uid'][i]) == uid
            X = data['full'][i, :, 4:12].astype(np.float64)
            a, b = data['retinal_mask'][i, 1:3]
            pooled = data['pooled'][i, 1:3, 4:12]
        p, i = seeds[uid]
        with np.load(p, allow_pickle=False) as data:
            assert int(data['uid'][i]) == uid
            w = data['seed'][i, 4:12].astype(np.float64)
        for i, m in enumerate((a, b)):
            assert np.array_equal(np.flatnonzero(m), meta['current_masks'][i])
            assert np.array_equal(X[m].mean(0).astype('float32'), pooled[i])
        g, r = (a, b) if good == 0 else (b, a)
        s = g & r
        assert (int(a.sum()), int(b.sum()), int(s.sum())) == tuple(int(z[k]) for k in ['n_a','n_b','shared'])
        cells = X @ w
        matrix = {}
        for name, mask in [('shared',s),('right_all',g),('right_only',g & ~r),('wrong_all',r),('wrong_only',r & ~g)]:
            assert mask.any()
            mean = X[mask].mean(axis=0)
            parts = mean*w
            val = float(mean@w)
            worst = max(worst, abs(val - cells[mask].mean()))
            matrix[name] = dict(n=int(mask.sum()), score=val, mean_normalized_signal=mean.tolist(),
                                weighted_channels=parts.tolist(), T4=float(parts[:4].sum()), T5=float(parts[4:].sum()))
        out = {}
        for label in ['right','wrong']:
            whole, own, shared = (matrix[n] for n in (label+'_all', label+'_only', 'shared'))
            frac = shared['n']/whole['n']
            assert shared['n'] + own['n'] == whole['n']
            worst = max(worst,abs(whole['score']-(frac*shared['score']+(1-frac)*own['score'])))
            out[label] = dict(shared_fraction=frac,score_change=own['score']-whole['score'])
        gap = {}
        for label, prefix in [('all','whole'),('only','exclusive')]:
            right, wrong = matrix['right_'+label], matrix['wrong_'+label]
            gap[label] = {k:right[k]-wrong[k] for k in ('score','T4','T5')}
            worst = max(worst,abs(gap[label]['score']-side*float(z[prefix+'_margin'])))
        delta = {k:gap['only'][k]-gap['all'][k] for k in gap['all']}
        worst = max(worst, abs(delta['T4']+delta['T5']-delta['score']))
        assert gap['all']['score'] > 1e-10 and gap['only']['score'] < -1e-10 and worst < 1e-10
        records.append(dict(uid=uid, video=v, cached_frame=meta['current_frame'], weight=w.tolist(), regions=matrix,
                            removal=out, target_minus_alternative=gap, change_in_gap=delta))
    for p, h in bound.items():
        assert sha(Path(p))==h
    result = dict(analysis='Retrospective decomposition of all five changed labeled primary pairs; conditional, not a new performance test.',
                  channels=CH, cases=records, selected_from=len(primary), max_absolute_verification_error=worst,
                  verified=['stored candidate labels/masks/counts','weighted means vs per-cell scores','mixture of overlap and exclusive cells',
                            'original signed margins for both policies', 'T4 plus T5 equals total margin shift'],
                  source_git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  analysis_source_sha256=sha(Path(__file__)),inputs_sha256=bound)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'decomposition.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    for z in records:
        a=z['regions'];print(z['uid'],z['cached_frame'],[(k,a[k]['n'],round(a[k]['score'],7)) for k in a],
             'delta T4/T5',round(z['change_in_gap']['T4'],7),round(z['change_in_gap']['T5'],7))
    print('max_error',worst)


if __name__ == '__main__':
    review()
