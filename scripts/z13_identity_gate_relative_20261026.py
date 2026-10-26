"""Two-stage FlyVis readout v3: within-query relative comparison in the target-identity stage.

Identical to z13_identity_gate_tagged_v1 (folds, models, gate, policy, controls, criteria) except
stage-1 inputs gain within-query comparisons: for each key scalar, the region's value minus the
best value among the other real regions of the same query, and its within-query rank. A per-region
pointwise model cannot otherwise compare a region with its neighbours, while choosing the target is
a relative decision. Features are read from the cached v2 arrays (no new FlyVis activity).
"""
from pathlib import Path
from datetime import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '3'
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_continue import paired
from scripts import z13_identity_gate_readout_20261026 as v1
from scripts import z13_identity_gate_tagged_20261026 as v2

IDENT = 'z13_identity_gate_relative_v1'
RUN = ROOT / 'runs/z13_identity_gate_relative_20261026'
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')
# F1 layout (v2): geo8 0:8 | x 8:76 | seed 76:144 | dot 144 | cos 145 | spatial 146:225 (summaries 146:157) | temporal 225:232 | tag 232:238
FLY_KEYS = [144, 145, 146, 147, 148, 149, 227, 228, 229, 231, 232, 233, 235] + list(range(8))
NOFLY_KEYS = list(range(18))  # geometry8 + tagged geometry10 in the no-fly F1


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(v2.RUN / 'protocol.json')
    sources = [Path(__file__), ROOT / 'scripts/z13_identity_gate_readout_20261026.py', ROOT / 'scripts/z13_identity_gate_tagged_20261026.py', v2.RUN / 'protocol.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'arms', 'stage2', 'secondary_policy', 'probe', 'threshold_grid', 'metric', 'internal_criterion', 'cannot_claim')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(),
        authorization='Автор 26.10.2026: «Пытайся реализовать модуль до 7:40» (автономная ночь).',
        change_from_v2='Only stage-1 inputs: within-query relative features (value minus best other real region; within-query rank) for key scalars. FlyVis arms: target response, cosine, spatial response mean/max/std/top, past-observation cosines and responses, tagged-memory cosine/distance/response difference, geometry8. No-fly arm: the same relative transform of geometry8 + tagged geometry. Cached v2 features with SHA256 check.',
        final='Not in this run: if internal passes, train26 evaluation is a separate registered step.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def relative(F, mask, keys):
    real = mask.copy(); real[:, 0] = False; K = F.shape[1]; out = []
    for c in keys:
        v = np.where(real, np.nan_to_num(F[..., c], nan=-1e9), -np.inf)
        best_other = np.empty_like(v)
        for k in range(K):
            q = v.copy(); q[:, k] = -np.inf; best_other[:, k] = q.max(1)
        diff = np.where(np.isfinite(best_other) & real & np.isfinite(F[..., c]), F[..., c] - best_other, np.nan)
        rank = np.where(real, (-v).argsort(1).argsort(1), np.nan)
        out += [diff, rank]
    return np.stack(out, -1).astype(np.float32)


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = v2.policy_net
    ent = v1.entries(); videos = p['fit84'] + p['fit20']; data = {}
    import torch
    torch.set_num_threads(3)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load
    cache_sha = {}
    for arm in v1.FLY:
        status('load', arm=arm); parts = []
        for split in ('fit84', 'fit20'):
            d = {k: t.numpy() for k, t in load(torch, namespace(), arm, split, ent).items()}; d['name'] = np.array(p[split])[d['video']]; parts.append(d)
        d = v1.merge(parts); f = v2.RUN / f'features_{arm}.npz'; cache_sha[f.relative_to(ROOT).as_posix()] = sha(f)
        with np.load(f) as z:
            assert np.array_equal(z['uid'], d['uid'])
            d['F1'] = np.concatenate([z['F1'], relative(z['F1'], d['mask'], FLY_KEYS)], -1); d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
        data[arm] = d
    base = data['D']
    nf = dict(base); G = np.concatenate([base['F1'][..., :8], base['Gtag']], -1); nf['F1'] = np.concatenate([G, relative(G, base['mask'], NOFLY_KEYS)], -1)
    data['nofly'] = nf; data['true_identity'] = base
    original = v1.stage1

    def stage1(d, arm, train_v, other_v):
        return original(dict(d), 'D', train_v, other_v) if arm == 'nofly' else original(d, arm, train_v, other_v)
    v1.stage1 = stage1
    internal = {a: dict(gate={}, identity={}, thresholds=dict(gate=[], identity=[]), counts=dict(gate=[], identity=[]), id_scores=[], id_labels=[]) for a in v1.ARMS}
    for f, split in enumerate(p['folds']):
        for arm in v1.ARMS:
            status('internal', fold=f, arm=arm)
            r = v1.one_split(data[arm], arm, split['train'], split['calibration'], split['validation'])
            for name in ('gate', 'identity'):
                internal[arm][name].update(r[name]['per']); internal[arm]['thresholds'][name].append(r[name]['threshold']); internal[arm]['counts'][name].append(r[name]['counts'])
            internal[arm]['id_scores'].append(r['identity_val'][0]); internal[arm]['id_labels'].append(r['identity_val'][1])
    means = {f'{a}_{n}': internal[a][n] for a in v1.ARMS for n in ('gate', 'identity')}
    zero = {v: 0. for v in videos}
    comp = {'D_gate-baseline': paired(means['D_gate'], zero), 'D_gate-nofly_gate': paired(means['D_gate'], means['nofly_gate']),
            'D_gate-C_gate': paired(means['D_gate'], means['C_gate']), 'D_gate-R1_gate': paired(means['D_gate'], means['R1_gate']),
            'D_identity-baseline': paired(means['D_identity'], zero), 'nofly_gate-baseline': paired(means['nofly_gate'], zero),
            'true_identity_gate-baseline': paired(means['true_identity_gate'], zero)}
    passed = comp['D_gate-baseline']['mean_per_video'] >= .02 and all(comp[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate'))
    summary = {a: dict(identity=v1.identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values()))), mean_identity=float(np.mean(list(internal[a]['identity'].values())))) for a in v1.ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, admit_final=bool(passed), paired=comp, summary=summary, per_video=means, cache_sha256=cache_sha,
                                             seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', admit_final=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
