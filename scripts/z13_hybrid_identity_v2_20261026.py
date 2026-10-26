"""z13_hybrid_identity_v2: richer appearance memory in the base branch of the hybrid identity readout.

Identical to z13_hybrid_identity_v1 (folds, models, gate, policy, arms) except the base branch adds:
cosine to the exemplar snapped to the observed detection (z13_masa_exemplar_snap_20261026) and 32
component-wise products of region and snapped-exemplar embeddings after a fixed label-free PCA (fitted on
region embeddings of fit104 queries), so the model can weight appearance dimensions instead of one cosine.
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

IDENT = 'z13_hybrid_identity_v2'
RUN = ROOT / 'runs/z13_hybrid_identity_v2_20261026'
SNAP = ROOT / 'runs/z13_masa_exemplar_snap_20261026'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')
ARMS = h1.ARMS
PCA_DIM = 32


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(h1.RUN / 'protocol.json'); s = read(SNAP / 'report.json'); assert s['complete'] and not s['ground_truth_read']
    sources = [Path(__file__), ROOT / 'scripts/z13_hybrid_identity_20261026.py', SNAP / 'report.json', h1.MASA / 'report.json', h1.RUN / 'internal_report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'stage2', 'probe', 'threshold_grid', 'metric', 'arms', 'alpha', 'criteria', 'final', 'fly_branch')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(), pca_dim=PCA_DIM,
        authorization='Решение автора 26.10.2026 ~18:30 MSK (решить проблему узнавания любым способом); продолжение после z13_hybrid_identity_v1.',
        change_from_v1='Base branch only: + cosine to snapped exemplar + 32 products of PCA-projected region and snapped-exemplar embeddings (label-free PCA on fit104 region embeddings) + within-query relative version of the snapped cosine.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def snapped(d):
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    reg = np.zeros((N, K, 256), np.float32); valid = np.zeros((N, K), bool); ex = np.zeros((N, 256), np.float32)
    for v in np.unique(d['name']):
        with np.load(h1.MASA / 'videos' / (v + '.npz')) as z:
            for n, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is None: continue
                k = min(z['region'].shape[1], K - 1); reg[i, 1:1 + k] = z['region'][n, :k]; valid[i, 1:1 + k] = z['valid'][n, :k]
        with np.load(SNAP / 'videos' / (v + '.npz')) as z:
            for n, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is not None: ex[i] = z['exemplar'][n]
    sample = reg[valid]; mean = sample.mean(0); _, _, vt = np.linalg.svd(sample[::max(1, len(sample) // 20000)] - mean, full_matrices=False); P = vt[:PCA_DIM].T
    rp = (reg - mean) @ P; ep = (ex - mean) @ P
    cos = np.where(valid, (reg * ex[:, None]).sum(-1), np.nan)[..., None]
    prod = np.where(valid[..., None], rp * ep[:, None], np.nan)
    return np.concatenate([cos, prod], -1).astype(np.float32)


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = v2.policy_net; h1.status = status
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
    A = h1.appearance(base, previous); S = snapped(base)
    G = np.concatenate([base['F1'][..., :8], base['Gtag'], A, S[..., :1]], -1)
    B = np.concatenate([G, v3.relative(G, base['mask'], list(range(G.shape[-1]))), S[..., 1:]], -1)
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
    zero = {v: 0. for v in videos}; prev = read(h1.RUN / 'internal_report.json')['per_video']
    comp = {'base_gate-baseline': paired(means['base_gate'], zero), 'base_D_gate-baseline': paired(means['base_D_gate'], zero),
            'base_D_gate-base_gate': paired(means['base_D_gate'], means['base_gate']), 'base_D_gate-base_C_gate': paired(means['base_D_gate'], means['base_C_gate']),
            'base_D_gate-base_R1_gate': paired(means['base_D_gate'], means['base_R1_gate']), 'base_D_gate-v1_base_D_gate': paired(means['base_D_gate'], prev['base_D_gate']),
            'base_gate-v1_base_gate': paired(means['base_gate'], prev['base_gate']), 'true_identity_gate-baseline': paired(means['true_identity_gate'], zero)}
    best = max(('base_gate-baseline', 'base_D_gate-baseline'), key=lambda k: comp[k]['mean_per_video'])
    problem = comp[best]['mean_per_video'] >= .02 and comp[best]['ci95'][0] > 0
    summary = {a: dict(identity=v1.identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values())))) for a in ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, problem_pass=bool(problem), best=best, fly_contribution_pass=bool(comp['base_D_gate-base_gate']['ci95'][0] > 0),
                                             paired=comp, summary=summary, per_video=means, seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', problem_pass=bool(problem), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
