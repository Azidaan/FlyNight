"""z13_hybrid_fly_confirm_v1: one-time train26 check of the FlyVis contribution in the frozen hybrid identity v2.

Frozen configuration of z13_hybrid_identity_v2 (features, gradient-boosted models, gate, policy, arms). Models are
fit on fit84 (stage 1 nested), the threshold on fit20, and evaluated once on train26. Label-free PCA of the appearance
products is fit on fit104 regions only. train26 features are built with the same code as the cached fit104 ones.
Registered before any train26 number of this configuration was computed.
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
from scripts import z13_hybrid_identity_20261026 as h1
from scripts import z13_hybrid_identity_v2_20261026 as h2

IDENT = 'z13_hybrid_fly_confirm_v1'
RUN = ROOT / 'runs/z13_hybrid_fly_confirm_20261026'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')
ARMS = h1.ARMS


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(h2.RUN / 'protocol.json')
    sources = [Path(__file__), ROOT / 'scripts/z13_hybrid_identity_v2_20261026.py', ROOT / 'scripts/z13_hybrid_identity_20261026.py',
               ROOT / 'scripts/z13_identity_gate_readout_20261026.py', ROOT / 'scripts/z13_identity_gate_tagged_20261026.py', ROOT / 'scripts/z13_identity_gate_relative_20261026.py',
               h2.RUN / 'internal_report.json', h1.MASA / 'report.json', h2.SNAP / 'report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'stage2', 'probe', 'threshold_grid', 'metric', 'arms', 'alpha', 'pca_dim')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(),
        authorization='Продолжение работы по решению автора 26.10.2026 (решить проблему узнавания; вклад мухи — отдельно).',
        registered_before_train26=True, train26_status='reused by earlier z13 experiments, never used to choose this configuration',
        split=dict(train='fit84 (stage 1 nested inner folds)', calibration='fit20', evaluation='train26 once'),
        criteria=dict(primary_fly='base_D - base lower 95% bound > 0 AND base_D - base_R1 lower 95% bound > 0 on train26',
                      secondary=['base_D - base_C lower 95% > 0 (modernisation)', 'base_D - baseline >= +0.02 with lower 95% > 0 (problem)']),
        cannot_claim='Not HOTA. train26 is not an untouched test set. One evaluation; no retuning after viewing.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def snapped_fit(d, fit_rows):
    """h2.snapped with the PCA basis fitted on fit104 rows only."""
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    reg = np.zeros((N, K, 256), np.float32); valid = np.zeros((N, K), bool); ex = np.zeros((N, 256), np.float32)
    for v in np.unique(d['name']):
        with np.load(h1.MASA / 'videos' / (v + '.npz')) as z:
            for n, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is None: continue
                k = min(z['region'].shape[1], K - 1); reg[i, 1:1 + k] = z['region'][n, :k]; valid[i, 1:1 + k] = z['valid'][n, :k]
        with np.load(h2.SNAP / 'videos' / (v + '.npz')) as z:
            for n, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is not None: ex[i] = z['exemplar'][n]
    sample = reg[valid & fit_rows[:, None]]; mean = sample.mean(0)
    _, _, vt = np.linalg.svd(sample[::max(1, len(sample) // 20000)] - mean, full_matrices=False); P = vt[:h2.PCA_DIM].T
    rp = (reg - mean) @ P; ep = (ex - mean) @ P
    cos = np.where(valid, (reg * ex[:, None]).sum(-1), np.nan)[..., None]
    prod = np.where(valid[..., None], rp * ep[:, None], np.nan)
    return np.concatenate([cos, prod], -1).astype(np.float32)


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = v2.policy_net
    previous = read(v2.READINESS)['previous_query']
    ent = v1.entries(); points = np.asarray(read(v1.GEOMETRY)['points'], np.float64)
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
            assert np.array_equal(z['uid'], d['uid']); d['F1'] = z['F1']; d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
        status('train26 features', arm=arm)
        ev = v1.build(arm, 'train26', ent, points); v2.add_tag(ev, previous)
        keep = set(d) & set(ev); d = v1.merge([{k: d[k] for k in keep}, {k: ev[k] for k in keep}])
        d['F1'] = np.concatenate([d['F1'], v3.relative(d['F1'], d['mask'], v3.FLY_KEYS)], -1)
        np.savez_compressed(RUN / f'train26_features_{arm}.npz', uid=ev['uid'], F1=ev['F1'], Gtag=ev['Gtag'], ident=ev['ident'])
        fly[arm] = d
    base = fly['D']
    for arm in ('C', 'R1'): assert np.array_equal(fly[arm]['uid'], base['uid']) and np.array_equal(fly[arm]['ident'], base['ident'])
    status('appearance')
    fit_rows = np.isin(base['name'], p['fit84'] + p['fit20'])
    A = h1.appearance(base, previous); S = snapped_fit(base, fit_rows)
    G = np.concatenate([base['F1'][..., :8], base['Gtag'], A, S[..., :1]], -1)
    B = np.concatenate([G, v3.relative(G, base['mask'], list(range(G.shape[-1]))), S[..., 1:]], -1)
    data = {'base': dict(base, F1=B), 'true_identity': base}
    for arm in v1.FLY: data['base_' + arm] = dict(fly[arm], F1=np.concatenate([B, fly[arm]['F1'][..., 8:]], -1))
    original = v1.stage1

    def stage1(d, arm, train_v, other_v):
        return original(d, 'true_identity' if arm == 'true_identity' else 'D', train_v, other_v)
    v1.stage1 = stage1
    res = {}; counts = {}; thresholds = {}; idsum = {}
    for arm in ARMS:
        status('evaluate', arm=arm)
        r = v1.one_split(data[arm], arm, p['fit84'], p['fit20'], p['evaluate26'])
        res[arm] = r['gate']['per']; counts[arm] = r['gate']['counts']; thresholds[arm] = r['gate']['threshold']
        idsum[arm] = v1.identity_summary([r['identity_val'][0]], [r['identity_val'][1]])
    z26 = {v: 0. for v in p['evaluate26']}
    comp = {'base_D-base': paired(res['base_D'], res['base']), 'base_D-base_R1': paired(res['base_D'], res['base_R1']),
            'base_D-base_C': paired(res['base_D'], res['base_C']), 'base_D-baseline': paired(res['base_D'], z26),
            'base-baseline': paired(res['base'], z26), 'true_identity-baseline': paired(res['true_identity'], z26)}
    primary = comp['base_D-base']['ci95'][0] > 0 and comp['base_D-base_R1']['ci95'][0] > 0
    write(RUN / 'report.json', dict(complete=True, id=IDENT, primary_fly_pass=bool(primary), modernisation_pass=bool(comp['base_D-base_C']['ci95'][0] > 0),
                                    problem_pass=bool(comp['base_D-baseline']['mean_per_video'] >= .02 and comp['base_D-baseline']['ci95'][0] > 0),
                                    paired=comp, per_video=res, counts=counts, thresholds=thresholds, identity=idsum, train26_evaluations=1,
                                    seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', primary_fly_pass=bool(primary), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
