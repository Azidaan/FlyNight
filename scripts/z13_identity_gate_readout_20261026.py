"""Two-stage FlyVis readout: dense target identity of real regions, then a small action gate.

Stage 1 (FlyVis readout node) learns "is this real region our target" from offline identity
labels, available for many more regions than the rare helpful-action labels. Stage 2 decides
whether to transfer the full box of an alternative region, from the stage-1 probability and the
unchanged assignment context. Controls: the same two stages with C, with R1, with geometry-only
identity (no fly) and with true identity (ceiling only). Fixed gradient-boosted models, CPU.
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

IDENT = 'z13_identity_gate_readout_v1'
RUN = ROOT / 'runs/z13_identity_gate_readout_20261026'
TRIAL = ROOT / 'runs/z13_default_class_trial_20261025'
FOLLOW = ROOT / 'runs/z13_box_supervision_followup_20261025'
SPATIAL = ROOT / 'runs/z13_spatial_set_readout_20261025/recovery2/features'
TEMPORAL = ROOT / 'runs/z13_temporal_identity_20261025/features'
SUPERVISION = ROOT / 'runs/z13_temporal_pair_readiness_20261025/supervision'
GEOMETRY = ROOT / 'runs/z13_local_peak_20261021/geometry.json'
FLY = ('D', 'C', 'R1')
ARMS = ('D', 'C', 'R1', 'nofly', 'true_identity')
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')
PROBE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
             l2_regularization=1.0, early_stopping=False, random_state=0)
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
SPLIT_NAMES = dict(fit84='fit84', fit20='fit20', train26='evaluate26')


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    trial = read(TRIAL / 'protocol.json')
    sources = [Path(__file__), TRIAL / 'protocol.json', FOLLOW / 'cache/report.json', FOLLOW / 'training/report.json', GEOMETRY,
               ROOT / 'scripts/z13_default_class_trial_20261025.py', ROOT / 'scripts/z13_region_assignment_recovery2_20261025.py']
    write(RUN / 'protocol.json', dict(
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle', deadline=DEADLINE.isoformat(),
        authorization='Автор 26.10.2026 01:4x MSK: «до 7:40 GPU вся твоя. Можешь попробовать исправить этот модуль? В автономном режиме».',
        fit84=trial['fit84'], fit20=trial['fit20'], evaluate26=trial['evaluate26'], folds=trial['folds'], arms=list(ARMS),
        motivation='Read-only count 26.10 on fit104: helpful full-box alternatives are the true target region in 1196 of 1218 helpful queries; always transferring to the true target when it is a non-baseline alternative gives +2.87 pp [2.30; 3.44] mean-per-video net. Previous heads trained only on rare helpful/harmful labels.',
        stage1=dict(node='FlyVis readout: P(real region is our target)', labels='offline identity: current region identity == track target identity; unknown identities excluded; NULL excluded',
                    features='geometry8 (no assignment context) + pooled x(68) + target descriptor seed(68) + x.seed + cosine + per-column spatial summaries(79) + past-observation summaries(7) of the same arm',
                    nofly='geometry8 only', true_identity='ceiling: probability = true label (unknown = NaN)', fitting='nested: inner 4 video folds give out-of-fold probabilities on outer-train videos; full outer-train fit for calibration/validation'),
        stage2=dict(features='geometry8 + assignment context6 + stage-1 probability of this region, max probability of other regions, probability of the baseline-chosen region, rank, margin',
                    labels='helpful=1 vs harmful=0 alternatives (non-baseline actions); neutral excluded from fitting',
                    policy='argmax gate probability over alternatives; act if >= threshold chosen on calibration videos (max mean-per-video net, ties larger threshold)'),
        secondary_policy='identity only: argmax stage-1 probability over alternatives; threshold on calibration videos',
        probe=dict(model='sklearn HistGradientBoostingClassifier', **PROBE), threshold_grid=THRESHOLDS,
        metric='mean over videos of (helpful - harmful)/known queries; paired bootstrap 5000 over videos (scripts/z13_continue.paired)',
        internal_criterion='Primary D two-stage gate on fit104 outer validation: mean >= +0.02 AND positive lower 95% bounds for D-baseline(0), D-nofly, D-C, D-R1.',
        final='Only if internal passes: stage models fit on fit84 (nested), threshold on fit20, single evaluation on reused train26 with the same criteria plus D-previous D_context (z13_box_supervision_followup_v1). No HOTA in this run.',
        cannot_claim='Not HOTA. fit104 and train26 were reused by earlier trials. Fixed boosted models are a readout implementation, not a probe of all possible information. true_identity is a ceiling, not a model.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def entries():
    out = {}
    for v, h in read(FOLLOW / 'cache/report.json')['manifest'].items():
        f = FOLLOW / 'cache/videos' / (v + '.json'); assert sha(f) == h
        out.update({int(k): x for k, x in read(f)['entries'].items()})
    return out


def chunks(folder):
    for marker in sorted(folder.glob('chunk_*.json')):
        path = marker.with_suffix('.npz'); assert sha(path) == read(marker)['sha256']
        yield path


def build(arm, split, ent, points):
    import torch
    torch.set_num_threads(3)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load
    p = read(RUN / 'protocol.json'); videos = p[SPLIT_NAMES[split]]
    d = {k: v.numpy() for k, v in load(torch, namespace(), arm, split, ent).items()}
    d['name'] = np.array(videos)[d['video']]
    N, K = d['mask'].shape; index = {int(u): i for i, u in enumerate(d['uid'])}
    S = np.full((N, K, 79), np.nan, np.float32); seen = 0
    for path in chunks(SPATIAL / arm / split):
        with np.load(path) as z:
            for r, u in enumerate(z['uid']):
                i = index[int(u)]; seed = d['seed'][i].astype(np.float64); seen += 1
                for k in range(min(K, z['retinal_mask'].shape[1])):
                    m = z['retinal_mask'][r, k]
                    if not m.any() or not d['mask'][i, k]: continue
                    act = z['full'][r][m].astype(np.float64)
                    assert np.abs(act.mean(0) - d['x'][i, k]).max() < 1e-4
                    resp = act @ seed; rel = (points[m] - z['centers'][r, k]) / np.maximum(z['scales'][r, k], 1e-6)
                    w = resp - resp.min(); w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                    quads = [resp[((rel[:, 0] > 0) == a) & ((rel[:, 1] > 0) == b)] for a in (False, True) for b in (False, True)]
                    top = np.sort(resp)[-max(1, len(resp) // 4):]
                    S[i, k, :11] = [resp.mean(), resp.max(), resp.std(), top.mean(), *(w @ rel), *[q.mean() if len(q) else resp.mean() for q in quads], len(resp)]
                    S[i, k, 11:] = act.std(0)
        status('features', arm=arm, split=split, rows_seen=seen, rows_total=N)
    assert seen == N, (arm, split, seen, N)
    T = np.full((N, K, 7), np.nan, np.float32); T[..., 0] = 0.
    for path in chunks(TEMPORAL / arm / split):
        with np.load(path) as z:
            for r, u in enumerate(z['uid']):
                i = index.get(int(u))
                if i is None or not z['previous_mask'][r].any(): continue
                prev = z['previous_x'][r][z['previous_mask'][r]].astype(np.float64); seed = d['seed'][i].astype(np.float64); presp = prev @ seed
                pn = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-9)
                for k in range(K):
                    if not d['mask'][i, k]: continue
                    x = d['x'][i, k].astype(np.float64); cos = pn @ (x / max(np.linalg.norm(x), 1e-9)); b = int(cos.argmax())
                    T[i, k] = [1., float(z['gap_frames'][r]), cos.max(), cos.mean(), presp[b], presp.max(), float(x @ seed) - presp.max()]
    x = d['x'].astype(np.float64); s = d['seed'].astype(np.float64)[:, None]; dot = (x * s).sum(-1)
    cos = dot / np.maximum(np.linalg.norm(x, axis=-1) * np.linalg.norm(s, axis=-1), 1e-9)
    d['F1'] = np.concatenate([d['geo'][..., :8], d['x'], np.repeat(d['seed'][:, None], K, 1), dot[..., None], cos[..., None], S, T], -1).astype(np.float32)
    ident = np.full((N, K), -1, np.int8)
    for v in np.unique(d['name']):
        for row in read(SUPERVISION / (v + '.json'))['rows']:
            i = index.get(int(row['uid']))
            if i is None: continue
            t = row['target_identity']
            for j, c in enumerate(row['current_identity']):
                if t and c: ident[i, j + 1] = int(c == t)
    d['ident'] = ident
    return d


def merge(parts):
    width = max(p['mask'].shape[1] for p in parts); out = {}
    for k in parts[0]:
        if parts[0][k].ndim >= 2 and k != 'seed':
            fill = -1 if k == 'ident' else 0
            out[k] = np.concatenate([np.pad(p[k], [(0, 0), (0, width - p[k].shape[1])] + [(0, 0)] * (p[k].ndim - 2), constant_values=fill) for p in parts])
        else: out[k] = np.concatenate([p[k] for p in parts])
    assert len(set(out['uid'].tolist())) == len(out['uid'])
    return out


def fit_predict(Xtr, ytr, Xte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**PROBE).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def stage1(d, arm, train_v, other_v):
    """Out-of-fold identity probabilities on train videos (inner 4 folds) and full-fit on other videos."""
    real = d['mask'].copy(); real[:, 0] = False
    if arm == 'true_identity':
        return np.where(d['ident'] >= 0, d['ident'], np.nan).astype(np.float64) * np.where(real, 1, np.nan)
    X = d['F1'] if arm in FLY else d['F1'][..., :8]
    prob = np.full(d['mask'].shape, np.nan); lab = real & (d['ident'] >= 0)
    inner = {v: i % 4 for i, v in enumerate(sorted(train_v))}
    for f in range(4):
        fit_v = [v for v in train_v if inner[v] != f]; out_v = [v for v in train_v if inner[v] == f]
        tr = np.isin(d['name'], fit_v)[:, None] & lab; te = np.isin(d['name'], out_v)[:, None] & real
        prob[te] = fit_predict(X[tr], d['ident'][tr], X[te])
    tr = np.isin(d['name'], train_v)[:, None] & lab; te = np.isin(d['name'], other_v)[:, None] & real
    prob[te] = fit_predict(X[tr], d['ident'][tr], X[te])
    return prob


def gate_features(d, prob):
    K = prob.shape[1]; p = np.nan_to_num(prob, nan=-1.)
    p_real = np.where(d['mask'], p, -1.); best_other = np.empty_like(p)
    for k in range(K):
        q = p_real.copy(); q[:, k] = -1.; best_other[:, k] = q.max(1)
    base = np.where(d['geo'][..., -1] > 0.5, p, -np.inf).max(1); base = np.where(np.isfinite(base), base, -1.)
    rank = (-p_real).argsort(1).argsort(1).astype(np.float32)
    return np.concatenate([d['geo'], prob[..., None], best_other[..., None], np.repeat(base[:, None, None], K, 1), rank[..., None], (p - best_other)[..., None]], -1)


def policy_net(d, score, videos, t):
    known = d['kind'] >= 0; alt = d['mask'] & (d['geo'][..., -1] < 0.5) & known[:, None]
    s = np.where(alt, np.nan_to_num(score, nan=-1.), -1.); k = s.argmax(1); best = s[np.arange(len(s)), k]
    act = known & (best >= t); ar = np.arange(len(s))
    net = np.where(act, d['good'][ar, k].astype(int) - d['harm'][ar, k].astype(int), 0)
    per = {v: float(net[(d['name'] == v) & known].mean()) for v in videos if ((d['name'] == v) & known).any()}
    return per, dict(selected=int(act.sum()), beneficial=int((net > 0).sum()), harmful=int((net < 0).sum()))


def one_split(d, arm, train_v, cal_v, val_v):
    prob = stage1(d, arm, train_v, cal_v + val_v); G = gate_features(d, prob)
    known = d['kind'] >= 0; alt = d['mask'] & (d['geo'][..., -1] < 0.5) & known[:, None]
    tr = np.isin(d['name'], train_v)[:, None] & alt & (d['good'] | d['harm'])
    te = np.isin(d['name'], cal_v + val_v)[:, None] & alt
    gate = np.full(prob.shape, np.nan); gate[te] = fit_predict(G[tr], d['good'][tr].astype(int), G[te])
    out = {}
    for name, score in (('gate', gate), ('identity', prob)):
        grid = [(float(np.mean(list(policy_net(d, score, cal_v, t)[0].values()))), t) for t in THRESHOLDS]
        t = max(grid)[1]; per, counts = policy_net(d, score, val_v, t); out[name] = dict(per=per, threshold=t, counts=counts)
    real = d['mask'].copy(); real[:, 0] = False; lab = np.isin(d['name'], val_v)[:, None] & real & (d['ident'] >= 0)
    out['identity_val'] = (prob[lab], d['ident'][lab])
    return out


def identity_summary(scores, labels):
    from sklearn.metrics import roc_auc_score
    s = np.concatenate(scores); y = np.concatenate(labels); ok = np.isfinite(s)
    return dict(auc=float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok])) == 2 else None, n=int(ok.sum()), positives=int(y[ok].sum()))


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    ent = entries(); points = np.asarray(read(GEOMETRY)['points'], np.float64); videos = p['fit84'] + p['fit20']
    data = {}
    for arm in FLY:
        status('features', arm=arm)
        data[arm] = merge([build(arm, 'fit84', ent, points), build(arm, 'fit20', ent, points)])
    base = data['D']
    for arm in ('C', 'R1'):
        assert np.array_equal(data[arm]['uid'], base['uid']) and np.array_equal(data[arm]['good'], base['good']) and np.array_equal(data[arm]['ident'], base['ident'])
    data['nofly'] = base; data['true_identity'] = base
    internal = {a: dict(gate={}, identity={}, thresholds=dict(gate=[], identity=[]), counts=dict(gate=[], identity=[]), id_scores=[], id_labels=[]) for a in ARMS}
    for f, split in enumerate(p['folds']):
        for arm in ARMS:
            status('internal', fold=f, arm=arm)
            r = one_split(data[arm], arm, split['train'], split['calibration'], split['validation'])
            for name in ('gate', 'identity'):
                internal[arm][name].update(r[name]['per']); internal[arm]['thresholds'][name].append(r[name]['threshold']); internal[arm]['counts'][name].append(r[name]['counts'])
            internal[arm]['id_scores'].append(r['identity_val'][0]); internal[arm]['id_labels'].append(r['identity_val'][1])
    means = {f'{a}_{n}': internal[a][n] for a in ARMS for n in ('gate', 'identity')}
    zero = {v: 0. for v in videos}
    comp = {'D_gate-baseline': paired(means['D_gate'], zero), 'D_gate-nofly_gate': paired(means['D_gate'], means['nofly_gate']),
            'D_gate-C_gate': paired(means['D_gate'], means['C_gate']), 'D_gate-R1_gate': paired(means['D_gate'], means['R1_gate']),
            'D_identity-baseline': paired(means['D_identity'], zero), 'D_identity-nofly_identity': paired(means['D_identity'], means['nofly_identity']),
            'true_identity_gate-baseline': paired(means['true_identity_gate'], zero), 'true_identity_identity-baseline': paired(means['true_identity_identity'], zero),
            'nofly_gate-baseline': paired(means['nofly_gate'], zero)}
    passed = comp['D_gate-baseline']['mean_per_video'] >= .02 and all(comp[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate'))
    summary = {a: dict(identity=identity_summary(internal[a]['id_scores'], internal[a]['id_labels']), thresholds=internal[a]['thresholds'],
                       counts={n: {k: sum(c[k] for c in internal[a]['counts'][n]) for k in ('selected', 'beneficial', 'harmful')} for n in ('gate', 'identity')},
                       mean_gate=float(np.mean(list(internal[a]['gate'].values()))), mean_identity=float(np.mean(list(internal[a]['identity'].values())))) for a in ARMS}
    write(RUN / 'internal_report.json', dict(complete=True, id=IDENT, admit_final=bool(passed), paired=comp, summary=summary, per_video=means,
                                             seconds=time.monotonic() - began, cuda_initialized=False, hota_measured=False, new_train26_evaluations=0))
    if passed:
        status('final')
        ev = {}
        for arm in FLY:
            ev[arm] = merge([data[arm], build(arm, 'train26', ent, points)])
        ev['nofly'] = ev['D']; ev['true_identity'] = ev['D']; fm = {}
        for arm in ARMS:
            r = one_split(ev[arm], arm, p['fit84'], p['fit20'], p['evaluate26']); fm[arm] = r['gate']['per']; fm[arm + '_counts'] = r['gate']['counts']
        old = read(FOLLOW / 'training/report.json')['mean_seed_per_video']['D_context']; z26 = {v: 0. for v in p['evaluate26']}
        fc = {'D_gate-baseline': paired(fm['D'], z26), 'D_gate-nofly_gate': paired(fm['D'], fm['nofly']), 'D_gate-C_gate': paired(fm['D'], fm['C']),
              'D_gate-R1_gate': paired(fm['D'], fm['R1']), 'D_gate-previous_D_context': paired(fm['D'], old), 'true_identity_gate-baseline': paired(fm['true_identity'], z26)}
        fpass = fc['D_gate-baseline']['mean_per_video'] >= .02 and all(fc[k]['ci95'][0] > 0 for k in ('D_gate-baseline', 'D_gate-nofly_gate', 'D_gate-C_gate', 'D_gate-R1_gate', 'D_gate-previous_D_context'))
        write(RUN / 'final_report.json', dict(complete=True, primary_pass=bool(fpass), paired=fc, per_video={a: fm[a] for a in ARMS}, counts={a: fm[a + '_counts'] for a in ARMS},
                                              train26_evaluations=1, reused_train26=True, hota_measured=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', admit_final=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
