"""Stage B of z13_hybrid_identity_v1: appearance base branch + FlyVis branch for target identity, then transfer gate.

Same two stages, folds, gradient-boosted models, gate and policy as z13_identity_gate_relative_v1. Identity
inputs per real region: BASE = geometry8 + tagged geometry10 + MASA appearance similarities (to the
observed exemplar, to the tagged previous target region, to an accumulated tagged bank, alpha 0.5) and their
within-query relative versions. BASE+X adds the v3 FlyVis features of arm X (D, C or R1). true_identity is
the ceiling. The FlyVis contribution is BASE+D minus BASE; modernisation is BASE+D minus BASE+C.
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
from scripts import z13_identity_gate_relative_20261026 as v3

IDENT = 'z13_hybrid_identity_v1'
RUN = ROOT / 'runs/z13_hybrid_identity_20261026'
MASA = ROOT / 'runs/z13_masa_regions_20261026/recovery1'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')
ARMS = ('base', 'base_D', 'base_C', 'base_R1', 'true_identity')
ALPHA = 0.5


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(v3.RUN / 'protocol.json'); m = read(MASA / 'report.json'); assert m['complete'] and not m['ground_truth_read']
    sources = [Path(__file__), ROOT / 'scripts/z13_identity_gate_readout_20261026.py', ROOT / 'scripts/z13_identity_gate_tagged_20261026.py',
               ROOT / 'scripts/z13_identity_gate_relative_20261026.py', v2.READINESS, MASA / 'report.json', v3.RUN / 'internal_report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'stage2', 'probe', 'threshold_grid', 'metric')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(), arms=list(ARMS), alpha=ALPHA,
        authorization='Решение автора 26.10.2026 ~18:30 MSK: решить проблему узнавания любым способом, допускается выход за пределы мухи (схема база + ветвь мухи).',
        base_branch='geometry8 + tagged geometry10 + MASA cosine to observed exemplar, to tagged previous target region, to accumulated tagged bank (alpha 0.5) + within-query relative versions (diff to best other region, rank)',
        fly_branch='v3 FlyVis stage-1 features of arm X (cached v2 features + v3 relative transform)',
        criteria=dict(problem='best of base / base_D: mean >= +0.02 and lower 95% > 0', fly='base_D - base lower 95% > 0', modernisation='base_D - base_C lower 95% > 0', eye='base_D - base_R1 lower 95% > 0'),
        final='Only if the problem criterion passes: single train26 evaluation (fit84 train, fit20 calibration) as a separate registered step.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def appearance(d, previous):
    """Per-region appearance similarities (exemplar, tagged previous target, accumulated bank), NaN for NULL and missing."""
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    reg = np.zeros((N, K, 256), np.float32); valid = np.zeros((N, K), bool); ex = np.zeros((N, 256), np.float32)
    for v in np.unique(d['name']):
        with np.load(MASA / 'videos' / (v + '.npz')) as z:
            for n, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is None: continue
                w = z['region'].shape[1]; k = min(w, K - 1)
                reg[i, 1:1 + k] = z['region'][n, :k]; valid[i, 1:1 + k] = z['valid'][n, :k]; ex[i] = z['exemplar'][n]
    assert (valid == (d['mask'] & (np.arange(K) > 0)[None])).all(), 'MASA regions must match readout regions'
    tag = np.full((N, 256), np.nan, np.float32)
    for i in range(N):
        b = np.nonzero(d['mask'][i] & (d['geo'][i, :, -1] > 0.5))[0]
        if len(b) and valid[i, b[0]]: tag[i] = reg[i, b[0]]
    prev_tag = np.full((N, 256), np.nan, np.float32); bank = np.full((N, 256), np.nan, np.float32)
    order = np.argsort(d['uid'])
    for i in order:
        pq = previous.get(str(int(d['uid'][i]))); j = index.get(int(pq['previous_uid'])) if pq else None
        if j is None: continue
        prev_tag[i] = tag[j]
        b = bank[j]
        if np.isfinite(tag[j]).all(): b = tag[j] if not np.isfinite(b).all() else ALPHA * tag[j] + (1 - ALPHA) * b
        bank[i] = b
    out = np.full((N, K, 3), np.nan, np.float32)
    for c, ref in enumerate((ex, prev_tag, bank)):
        refn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
        s = (reg * refn[:, None]).sum(-1)
        out[..., c] = np.where(valid & np.isfinite(refn).all(1)[:, None], s, np.nan)
    return out


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = v2.policy_net
    previous = read(v2.READINESS)['previous_query']
    ent = v1.entries(); videos = p['fit84'] + p['fit20']; data = {}
    import torch
    torch.set_num_threads(3)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load
    fly = {}
    for arm in v1.FLY:
        status('load', arm=arm); parts = []
        for split in ('fit84', 'fit20'):
            d = {k: t.numpy() for k, t in load(torch, namespace(), arm, split, ent).items()}; d['name'] = np.array(p[split])[d['video']]; parts.append(d)
        d = v1.merge(parts)
        with np.load(v2.RUN / f'features_{arm}.npz') as z:
            assert np.array_equal(z['uid'], d['uid'])
            d['F1'] = np.concatenate([z['F1'], v3.relative(z['F1'], d['mask'], v3.FLY_KEYS)], -1); d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
        fly[arm] = d
    base = fly['D']; status('appearance')
    A = appearance(base, previous)
    G = np.concatenate([base['F1'][..., :8], base['Gtag'], A], -1)
    B = np.concatenate([G, v3.relative(G, base['mask'], list(range(G.shape[-1])))], -1)
    data['base'] = dict(base, F1=B)
    for arm in v1.FLY:
        data['base_' + arm] = dict(fly[arm], F1=np.concatenate([B, fly[arm]['F1'][..., 8:]], -1))
    data['true_identity'] = base
    original = v1.stage1

    def stage1(d, arm, train_v, other_v):
        return original(d, 'true_identity' if arm == 'true_identity' else 'D', train_v, other_v)
    v1.stage1 = stage1
    internal = {a: dict(gate={}, identity={}, thresholds=dict(gate=[], identity=[]), counts=dict(gate=[], identity=[]), id_scores=[], id_labels=[]) for a in ARMS}
    for f, split in enumerate(p['folds']):
        for arm in ARMS:
            status('internal', fold=f, arm=arm)
            r = v1.one_split(data[arm], arm, split['train'], split['calibration'], split['validation'])
            for name in ('gate', 'identity'):
                internal[arm][name].update(r[name]['per']); internal[arm]['thresholds'][name].append(r[name]['threshold']); internal[arm]['counts'][name].append(r[name]['counts'])
            internal[arm]['id_scores'].append(r['identity_val'][0]); internal[arm]['id_labels'].append(r['identity_val'][1])
    means = {f'{a}_{n}': internal[a][n] for a in ARMS for n in ('gate', 'identity')}
    zero = {v: 0. for v in videos}; v3res = read(v3.RUN / 'internal_report.json')['per_video']
    comp = {'base_gate-baseline': paired(means['base_gate'], zero), 'base_D_gate-baseline': paired(means['base_D_gate'], zero),
            'base_D_gate-base_gate': paired(means['base_D_gate'], means['base_gate']), 'base_D_gate-base_C_gate': paired(means['base_D_gate'], means['base_C_gate']),
            'base_D_gate-base_R1_gate': paired(means['base_D_gate'], means['base_R1_gate']), 'base_gate-v3_D_gate': paired(means['base_gate'], v3res['D_gate']),
            'true_identity_gate-baseline': paired(means['true_identity_gate'], zero)}
    best = max(('base_gate-baseline', 'base_D_gate-baseline'), key=lambda k: comp[k]['mean_per_video'])
    problem = comp[best]['mean_per_video'] >= .02 and comp[best]['ci95'][0] > 0
    fly_pass = comp['base_D_gate-base_gate']['ci95'][0] > 0
    summary = {a: dict(identity=v1.identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values())))) for a in ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, problem_pass=bool(problem), best=best, fly_contribution_pass=bool(fly_pass), paired=comp,
                                             summary=summary, per_video=means, seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', problem_pass=bool(problem), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
