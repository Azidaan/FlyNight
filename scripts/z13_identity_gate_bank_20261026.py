"""Two-stage FlyVis readout v4: accumulated observable target appearance of the track.

Identical to z13_identity_gate_relative_v1 except stage-1 inputs gain an accumulated memory of the
track's target: an exponential average (fixed alpha 0.5, set before the run) of the FlyVis activity
of every region the tracker itself assigned to the same observed track ID at its strictly earlier
queries (observable tags only). Features: cosine and distance of each current region to the bank,
the bank's target-model response, the current-minus-bank response, number of tagged observations,
and their within-query relative versions. The no-fly control receives the same bank built from the
tagged geometry. Cached v2 features are reused (SHA256 checked).
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

IDENT = 'z13_identity_gate_bank_v1'
RUN = ROOT / 'runs/z13_identity_gate_bank_20261026'
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')
ALPHA = 0.5


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(v3.RUN / 'protocol.json')
    sources = [Path(__file__), ROOT / 'scripts/z13_identity_gate_readout_20261026.py', ROOT / 'scripts/z13_identity_gate_tagged_20261026.py',
               ROOT / 'scripts/z13_identity_gate_relative_20261026.py', v2.READINESS, v3.RUN / 'protocol.json', v3.RUN / 'internal_report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'arms', 'stage2', 'secondary_policy', 'probe', 'threshold_grid', 'metric', 'internal_criterion', 'cannot_claim', 'final')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(), alpha=ALPHA,
        authorization='Автор 26.10.2026: «Пытайся реализовать модуль до 7:40» (автономная ночь).',
        change_from_v3='Only stage-1 inputs: accumulated tagged target bank (exponential average, alpha 0.5 fixed before run) over all strictly earlier tagged queries of the same observed track ID; cosine, distance, bank response, current-minus-bank response, number of tagged observations, plus within-query relative versions. No-fly arm: bank of tagged geometry8 (current minus bank) with its relative versions.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def banks(d, previous):
    """Exponential averages of tagged FlyVis activity and tagged geometry along each track's query chain."""
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    tag_x = np.full((N, 68), np.nan); tag_g = np.full((N, 8), np.nan)
    for i in range(N):
        b = np.nonzero(d['mask'][i] & (d['geo'][i, :, -1] > 0.5))[0]
        if len(b): tag_x[i] = d['x'][i, b[0]]; tag_g[i] = d['geo'][i, b[0], :8]
    bank_x = np.full((N, 68), np.nan); bank_g = np.full((N, 8), np.nan); count = np.zeros(N); done = np.zeros(N, bool)

    def resolve(i):
        stack = []
        while not done[i]:
            stack.append(i); prev = previous.get(str(int(d['uid'][i])))
            j = index.get(int(prev['previous_uid'])) if prev else None
            if j is None or j in stack: done[i] = True; break
            i = j
        for i in reversed(stack):
            prev = previous.get(str(int(d['uid'][i]))); j = index.get(int(prev['previous_uid'])) if prev else None
            if j is not None and j != i and done[j]:
                bx, bg, c = bank_x[j], bank_g[j], count[j]
                if np.isfinite(tag_x[j]).all():
                    bx = tag_x[j] if c == 0 else ALPHA * tag_x[j] + (1 - ALPHA) * bx
                    bg = tag_g[j] if c == 0 else ALPHA * tag_g[j] + (1 - ALPHA) * bg
                    c = c + 1
                bank_x[i], bank_g[i], count[i] = bx, bg, c
            done[i] = True
    for i in range(N): resolve(i)
    return bank_x, bank_g, count


def bank_features(d, bank_x, bank_g, count):
    N, K = d['mask'].shape; fly = np.full((N, K, 5), np.nan, np.float32); geo = np.full((N, K, 9), np.nan, np.float32)
    seed = d['seed'].astype(np.float64); has = count > 0
    bresp = np.where(has, (np.nan_to_num(bank_x) * seed).sum(1), np.nan)
    for k in range(K):
        x = d['x'][:, k].astype(np.float64); ok = d['mask'][:, k] & has
        cos = (x * np.nan_to_num(bank_x)).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(np.nan_to_num(bank_x), axis=1), 1e-9)
        dist = np.linalg.norm(x - np.nan_to_num(bank_x), axis=1); resp = (x * seed).sum(1)
        fly[ok, k] = np.stack([cos, dist, bresp, resp - bresp, count], 1)[ok]
        geo[ok, k] = np.concatenate([d['geo'][:, k, :8] - np.nan_to_num(bank_g), count[:, None]], 1)[ok]
    return fly, geo


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = v2.policy_net
    previous = read(v2.READINESS)['previous_query']
    ent = v1.entries(); videos = p['fit84'] + p['fit20']; data = {}; cache_sha = {}
    import torch
    torch.set_num_threads(3)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load
    for arm in v1.FLY:
        status('load', arm=arm); parts = []; fl = []; gl = []
        for split in ('fit84', 'fit20'):
            d = {k: t.numpy() for k, t in load(torch, namespace(), arm, split, ent).items()}; d['name'] = np.array(p[split])[d['video']]
            fly, geo = bank_features(d, *banks(d, previous)); parts.append(dict(d, BF=fly, BG=geo))
        d = v1.merge(parts); f = v2.RUN / f'features_{arm}.npz'; cache_sha[f.relative_to(ROOT).as_posix()] = sha(f)
        with np.load(f) as z:
            assert np.array_equal(z['uid'], d['uid'])
            F = np.concatenate([z['F1'], d['BF']], -1); nb = z['F1'].shape[-1]
            keys = v3.FLY_KEYS + list(range(nb, nb + 4))
            d['F1'] = np.concatenate([F, v3.relative(F, d['mask'], keys)], -1); d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
        data[arm] = d
    base = data['D']
    G = np.concatenate([base['F1'][..., :8], base['Gtag'], base['BG']], -1)
    nf = dict(base); nf['F1'] = np.concatenate([G, v3.relative(G, base['mask'], list(range(G.shape[-1] - 1)))], -1)
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
    zero = {v: 0. for v in videos}; v3res = read(v3.RUN / 'internal_report.json')['per_video']
    comp = {'D_gate-baseline': paired(means['D_gate'], zero), 'D_gate-nofly_gate': paired(means['D_gate'], means['nofly_gate']),
            'D_gate-C_gate': paired(means['D_gate'], means['C_gate']), 'D_gate-R1_gate': paired(means['D_gate'], means['R1_gate']),
            'D_gate-v3_D_gate': paired(means['D_gate'], v3res['D_gate']), 'D_identity-baseline': paired(means['D_identity'], zero),
            'nofly_gate-baseline': paired(means['nofly_gate'], zero), 'true_identity_gate-baseline': paired(means['true_identity_gate'], zero)}
    passed = comp['D_gate-baseline']['mean_per_video'] >= .02 and all(comp[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate'))
    summary = {a: dict(identity=v1.identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values()))), mean_identity=float(np.mean(list(internal[a]['identity'].values())))) for a in v1.ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, admit_final=bool(passed), paired=comp, summary=summary, per_video=means, cache_sha256=cache_sha,
                                             banked_queries=int(sum((data['D']['BF'][..., 4] > 0).any(1))), seconds=time.monotonic() - began,
                                             cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', admit_final=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
