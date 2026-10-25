"""Two-stage FlyVis readout v2: target identity with observable tagged memory, then transfer gate.

Same two stages, folds, models, policy and controls as z13_identity_gate_readout_v1. The only
change is in stage 1 inputs: memory of the region the tracker itself assigned to this track at
its previous query (observable tag, never ground truth): similarity of each current region's
FlyVis activity to that tagged previous region, its target-model response and the frame gap.
The no-fly control receives the same tag as geometry only (offset of each current region from
the tagged previous box), so the memory itself is matched across arms.
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

IDENT = 'z13_identity_gate_tagged_v1'
RUN = ROOT / 'runs/z13_identity_gate_tagged_20261026'
READINESS = ROOT / 'runs/z13_box_supervision_followup_20261025/temporal_readiness.json'
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    base = read(v1.RUN / 'protocol.json')
    sources = [Path(__file__), ROOT / 'scripts/z13_identity_gate_readout_20261026.py', READINESS, v1.RUN / 'protocol.json', v1.RUN / 'internal_report.json']
    write(RUN / 'protocol.json', dict(
        {k: base[k] for k in ('fit84', 'fit20', 'evaluate26', 'folds', 'arms', 'stage2', 'secondary_policy', 'probe', 'threshold_grid', 'metric', 'internal_criterion', 'final')},
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(),
        authorization='Автор 26.10.2026: «Пытайся реализовать модуль до 7:40» (автономная ночь).',
        change_from_v1='Only stage-1 inputs: observable tagged memory = the region the tracker assigned to the same observed track ID at its previous query (temporal_readiness previous_query; never ground truth, never future). FlyVis arms: cosine and L2 distance of current region activity to tagged previous activity, tagged previous target response, current minus tagged response, gap frames, has_tag. No-fly arm: same tag as geometry (current region geometry minus tagged previous region geometry, gap, has_tag).',
        cannot_claim=base['cannot_claim'],
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def policy_net(d, score, videos, t):
    """v1 rule and per-video metric unchanged; counts restricted to the evaluated videos (v1 counted all rows)."""
    known = d['kind'] >= 0; alt = d['mask'] & (d['geo'][..., -1] < 0.5) & known[:, None]
    s = np.where(alt, np.nan_to_num(score, nan=-1.), -1.); k = s.argmax(1); best = s[np.arange(len(s)), k]
    act = known & (best >= t); ar = np.arange(len(s)); inside = np.isin(d['name'], videos)
    net = np.where(act, d['good'][ar, k].astype(int) - d['harm'][ar, k].astype(int), 0)
    per = {v: float(net[(d['name'] == v) & known].mean()) for v in videos if ((d['name'] == v) & known).any()}
    return per, dict(selected=int((act & inside).sum()), beneficial=int(((net > 0) & inside).sum()), harmful=int(((net < 0) & inside).sum()))


def add_tag(d, previous):
    """Append tagged-memory features to F1 (fly) and build geometry tag for the no-fly arm."""
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    fly = np.full((N, K, 6), np.nan, np.float32); geo = np.full((N, K, 10), np.nan, np.float32)
    fly[..., 5] = 0.; geo[..., 9] = 0.; tagged = 0
    for i, u in enumerate(d['uid']):
        prev = previous.get(str(int(u)))
        if prev is None: continue
        j = index.get(int(prev['previous_uid']))
        if j is None: continue
        b = np.nonzero(d['mask'][j] & (d['geo'][j, :, -1] > 0.5))[0]
        if not len(b): continue
        b = int(b[0]); tagged += 1
        px = d['x'][j, b].astype(np.float64); presp = float(px @ d['seed'][i].astype(np.float64))
        for k in range(K):
            if not d['mask'][i, k]: continue
            x = d['x'][i, k].astype(np.float64)
            fly[i, k] = [x @ px / max(np.linalg.norm(x) * np.linalg.norm(px), 1e-9), np.linalg.norm(x - px), presp,
                         float(x @ d['seed'][i].astype(np.float64)) - presp, prev['gap_frames'], 1.]
            geo[i, k] = [*(d['geo'][i, k, :8] - d['geo'][j, b, :8]), prev['gap_frames'], 1.]
    d['F1'] = np.concatenate([d['F1'], fly], -1); d['Gtag'] = geo
    return tagged


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    v1.RUN = RUN; v1.status = status; v1.DEADLINE = DEADLINE; v1.policy_net = policy_net
    previous = read(READINESS)['previous_query']
    ent = v1.entries(); points = np.asarray(read(v1.GEOMETRY)['points'], np.float64); videos = p['fit84'] + p['fit20']
    data = {}; tagged = {}
    for arm in v1.FLY:
        status('features', arm=arm); parts = []
        for split in ('fit84', 'fit20'):
            part = v1.build(arm, split, ent, points); tagged[f'{arm}_{split}'] = add_tag(part, previous); parts.append(part)
        data[arm] = v1.merge(parts)
        np.savez_compressed(RUN / f'features_{arm}.npz', uid=data[arm]['uid'], F1=data[arm]['F1'], Gtag=data[arm]['Gtag'], ident=data[arm]['ident'])
    base = data['D']
    for arm in ('C', 'R1'):
        assert np.array_equal(data[arm]['uid'], base['uid']) and np.array_equal(data[arm]['ident'], base['ident'])
    nofly = dict(base); nofly['F1'] = np.concatenate([base['F1'][..., :8], base['Gtag']], -1)
    data['nofly'] = nofly; data['true_identity'] = base
    original_stage1 = v1.stage1

    def stage1(d, arm, train_v, other_v):
        if arm == 'nofly': return original_stage1(dict(d), 'D', train_v, other_v)
        return original_stage1(d, arm, train_v, other_v)
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
    v1res = read(v1.ROOT / 'runs/z13_identity_gate_readout_20261026/internal_report.json')['per_video']
    comp = {'D_gate-baseline': paired(means['D_gate'], zero), 'D_gate-nofly_gate': paired(means['D_gate'], means['nofly_gate']),
            'D_gate-C_gate': paired(means['D_gate'], means['C_gate']), 'D_gate-R1_gate': paired(means['D_gate'], means['R1_gate']),
            'D_gate-v1_D_gate': paired(means['D_gate'], v1res['D_gate']), 'D_identity-baseline': paired(means['D_identity'], zero),
            'true_identity_gate-baseline': paired(means['true_identity_gate'], zero), 'nofly_gate-baseline': paired(means['nofly_gate'], zero)}
    passed = comp['D_gate-baseline']['mean_per_video'] >= .02 and all(comp[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate'))
    summary = {a: dict(identity=v1.identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values()))), mean_identity=float(np.mean(list(internal[a]['identity'].values())))) for a in v1.ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, admit_final=bool(passed), paired=comp, summary=summary, per_video=means, tagged_rows=tagged,
                                             seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    if passed:
        status('final'); ev = {}
        for arm in v1.FLY:
            part = v1.build(arm, 'train26', ent, points); tagged[f'{arm}_train26'] = add_tag(part, previous)
            ev[arm] = v1.merge([data[arm], part])
        nf = dict(ev['D']); nf['F1'] = np.concatenate([ev['D']['F1'][..., :8], ev['D']['Gtag']], -1)
        ev['nofly'] = nf; ev['true_identity'] = ev['D']; fm = {}; fc_counts = {}
        for arm in v1.ARMS:
            r = v1.one_split(ev[arm], arm, p['fit84'], p['fit20'], p['evaluate26']); fm[arm] = r['gate']['per']; fc_counts[arm] = r['gate']['counts']
        old = read(v1.FOLLOW / 'training/report.json')['mean_seed_per_video']['D_context']; z26 = {v: 0. for v in p['evaluate26']}
        fc = {'D_gate-baseline': paired(fm['D'], z26), 'D_gate-nofly_gate': paired(fm['D'], fm['nofly']), 'D_gate-C_gate': paired(fm['D'], fm['C']),
              'D_gate-R1_gate': paired(fm['D'], fm['R1']), 'D_gate-previous_D_context': paired(fm['D'], old), 'true_identity_gate-baseline': paired(fm['true_identity'], z26)}
        fpass = fc['D_gate-baseline']['mean_per_video'] >= .02 and all(fc[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate', 'D_gate-previous_D_context'))
        write(RUN / 'final_report.json', dict(complete=True, primary_pass=bool(fpass), paired=fc, per_video={a: fm[a] for a in v1.ARMS}, counts=fc_counts,
                                              train26_evaluations=1, reused_train26=True, hota_measured=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', admit_final=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
