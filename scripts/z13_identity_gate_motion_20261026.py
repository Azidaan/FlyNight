"""Two-stage FlyVis readout v5: motion-signature consistency from FlyVis T4/T5 in target identity.

Identical to z13_identity_gate_relative_v1 except stage-1 inputs gain a fly-specific identity cue:
each region's direction vector from the T4/T5 channels (energies a-d = T4 + T5, net_x = a - b,
net_y = c - d, magnitude) and its consistency with the direction vector of the region the tracker
assigned to the same observed track at its previous query (cosine, difference norm, magnitude
ratio), plus within-query relative versions. No-fly control as in v3 (geometry + tagged geometry).
Channel layout: 68 = 34 FlyVis output types (OUTPUT_CELL_TYPES order: T4a-d = 4..7, T5a-d = 8..11)
+ their 6-neighbour means. R1 has no T4/T5 channels; it receives the same transform of its own
channels 4..11 as a matched control.
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

IDENT = 'z13_identity_gate_motion_v1'
RUN = ROOT / 'runs/z13_identity_gate_motion_20261026'
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(v3.RUN / 'protocol.json')
    sources = [Path(__file__), ROOT / 'scripts/z13_identity_gate_readout_20261026.py', ROOT / 'scripts/z13_identity_gate_tagged_20261026.py',
               ROOT / 'scripts/z13_identity_gate_relative_20261026.py', ROOT / 'tracking/flyvis_activity.py', v2.READINESS, v3.RUN / 'internal_report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'arms', 'stage2', 'secondary_policy', 'probe', 'threshold_grid', 'metric', 'internal_criterion', 'cannot_claim', 'final')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(),
        authorization='Автор 26.10.2026: «Пытайся реализовать модуль до 7:40» (автономная ночь).',
        change_from_v3='Only stage-1 inputs: T4/T5 direction vector of each region (a-d energies T4+T5, net_x=a-b, net_y=c-d, magnitude) and its consistency with the tagged previous target region (cosine, difference norm, magnitude ratio), with within-query relative versions. Same transform of channels 4..11 for R1 as matched control. Last attempt of this readout family under rule 6.16; stop the family if it fails.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def motion(x):
    e = x[..., 4:8] + x[..., 8:12]; nx = e[..., 0] - e[..., 1]; ny = e[..., 2] - e[..., 3]
    return np.stack([nx, ny], -1), e


def motion_features(d, previous):
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    vec, e = motion(d['x'].astype(np.float64)); mag = np.linalg.norm(vec, axis=-1)
    out = np.full((N, K, 10), np.nan, np.float32)
    out[..., :2] = vec; out[..., 2] = mag; out[..., 3:7] = e
    for i, u in enumerate(d['uid']):
        prev = previous.get(str(int(u))); j = index.get(int(prev['previous_uid'])) if prev else None
        if j is None: continue
        b = np.nonzero(d['mask'][j] & (d['geo'][j, :, -1] > 0.5))[0]
        if not len(b): continue
        tv = vec[j, b[0]]; tm = max(np.linalg.norm(tv), 1e-9)
        for k in range(K):
            if not d['mask'][i, k]: continue
            out[i, k, 7] = float(vec[i, k] @ tv) / max(mag[i, k] * tm, 1e-9); out[i, k, 8] = float(np.linalg.norm(vec[i, k] - tv)); out[i, k, 9] = mag[i, k] / tm
    return out


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
        status('load', arm=arm); parts = []
        for split in ('fit84', 'fit20'):
            d = {k: t.numpy() for k, t in load(torch, namespace(), arm, split, ent).items()}; d['name'] = np.array(p[split])[d['video']]
            parts.append(dict(d, MF=motion_features(d, previous)))
        d = v1.merge(parts); f = v2.RUN / f'features_{arm}.npz'; cache_sha[f.relative_to(ROOT).as_posix()] = sha(f)
        with np.load(f) as z:
            assert np.array_equal(z['uid'], d['uid'])
            F = np.concatenate([z['F1'], d['MF']], -1); nb = z['F1'].shape[-1]
            keys = v3.FLY_KEYS + [nb + 2, nb + 7, nb + 8, nb + 9]
            d['F1'] = np.concatenate([F, v3.relative(F, d['mask'], keys)], -1); d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
        data[arm] = d
    base = data['D']
    G = np.concatenate([base['F1'][..., :8], base['Gtag']], -1)
    nf = dict(base); nf['F1'] = np.concatenate([G, v3.relative(G, base['mask'], v3.NOFLY_KEYS)], -1)
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
                                             seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', admit_final=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
