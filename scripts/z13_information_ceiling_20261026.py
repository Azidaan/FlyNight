"""Information ceiling of saved FlyVis readout features for the full-box action (no new heads).

Fixed gradient-boosted probes separate helpful from harmful alternative actions on the
same five video folds of fit104 used by z13_default_class_trial_v1. Feature sets:
geometry+assignment context (no fly), pooled activity, per-column spatial summaries and
past-observation summaries, each for D, C and R1. Probes measure available information;
they are not proposed readout heads.
"""
from pathlib import Path
from datetime import datetime, timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys
import json
import time
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now

IDENT = 'z13_information_ceiling_v1'
RUN = ROOT / 'runs/z13_information_ceiling_20261026'
TRIAL = ROOT / 'runs/z13_default_class_trial_20261025'
FOLLOW = ROOT / 'runs/z13_box_supervision_followup_20261025'
SPATIAL = ROOT / 'runs/z13_spatial_set_readout_20261025/recovery2/features'
TEMPORAL = ROOT / 'runs/z13_temporal_identity_20261025/features'
GEOMETRY = ROOT / 'runs/z13_local_peak_20261021/geometry.json'
ARMS = ('D', 'C', 'R1')
SETS = ('pooled', 'spatial', 'temporal')
DEADLINE = datetime.fromisoformat('2026-10-26T07:30:00+03:00')
BUDGET_SECONDS = 5400
PROBE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
             l2_regularization=1.0, early_stopping=False, random_state=0)
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]
N_BOOT = 2000
REFERENCE_AUC = 0.93


def status(stage, **kw):
    if now() >= DEADLINE: raise TimeoutError('Author weekday deadline 07:30 MSK reached')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists()
    trial = read(TRIAL / 'protocol.json')
    sources = [Path(__file__), TRIAL / 'protocol.json', FOLLOW / 'cache/report.json', GEOMETRY,
               ROOT / 'scripts/z13_default_class_trial_20261025.py', ROOT / 'scripts/z13_region_assignment_recovery2_20261025.py']
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN / 'protocol.json', dict(
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cpu', priority='idle',
        deadline=DEADLINE.isoformat(), budget_seconds=BUDGET_SECONDS,
        authorization='Автор 26.10.2026: зарегистрируй и выполни на CPU проверку потолка информации по уже сохранённым признакам против геометрии с контекстом, со скользящим разбиением на fit104, без новых голов.',
        videos=trial['fit84'] + trial['fit20'], folds=trial['folds'], arms=list(ARMS), feature_sets=['geometry'] + [a + '_' + s for a in ARMS for s in SETS],
        units='Known queries (kind>=0) of fit104; alternatives = real-region or explicit-unmatched actions whose assignment-context flag is_baseline_choice is 0. Probe label: helpful (good) = 1 versus harmful (harm) = 0; neutral alternatives excluded from fitting and AUC, kept in the policy where choosing them gives 0.',
        features=dict(
            geometry='geo: 8 geometry inputs + 6 assignment-context features of the unchanged first association (loader of z13_default_class_trial_v1).',
            pooled='geometry + region mean activity x (68) + observed target descriptor seed (68) + x.seed + cosine(x, seed).',
            spatial='pooled + from saved full per-column activity (721x68) inside each region support: target response map r = activity.seed -> mean, max, std, top-quartile mean, positive-mass centroid offset (2, scale-normalised), four quadrant means; per-channel within-region std (68); support size.',
            temporal='pooled + from saved latest strictly earlier supported observation (same alignment clock as z13_temporal_identity_v1): has_prior, gap_frames, max/mean cosine of x to previous regions, response of best-matching previous region, max previous response, current response minus max previous response. Missing history = NaN.'),
        probe=dict(model='sklearn HistGradientBoostingClassifier', **PROBE, note='Fixed before run; no tuning; one deterministic fit per fold and feature set.'),
        policy=dict(rule='per query argmax probe probability over alternatives; act if >= threshold', threshold_grid=THRESHOLDS,
                    threshold_selection='fold calibration videos only: max mean-per-video net, ties -> larger threshold', metric='mean over videos of (helpful - harmful)/known queries, as in previous z13 readout trials'),
        statistics=dict(auc='pooled out-of-fold AUC over outer validation predictions of all 104 videos', bootstrap=f'{N_BOOT} paired video resamples, seed 0, 95% percentile intervals for AUC and net differences'),
        reference=dict(auc=REFERENCE_AUC, origin='research_flyvis_advantage_readout_2026-10-25/СВОДКА.md 1.5: binormal d prime 2.08 needed for +2 pp on fit84 base rates; descriptive reference, not a universal limit'),
        decision=dict(
            fly_information='For a feature set S: lower 95% bound of AUC(D_S) - AUC(geometry) > 0 AND lower bound of net(D_S) - net(geometry) > 0.',
            sufficient='Point AUC(D_S) >= 0.93 AND net(D_S) >= +0.02 with positive lower bound.',
            fly_specific='Lower bounds of AUC(D_S) - AUC(R1_S) > 0 and AUC(D_S) - AUC(C_S) > 0.',
            outcomes=['No S passes fly_information -> stop new readout heads on these saved features and these consequence labels; next change must alter representation input (true exemplar-vs-search spatial matching), labels or the action.',
                      'fly_information passes but not sufficient -> information exists but is insufficient for +2 pp with this action; do not train new heads of the same family.',
                      'fly_information and sufficient pass -> previous heads failed to use available information; one new head may be registered separately.']),
        cannot_claim='A probe gives a lower bound on information: a negative result does not prove that no learner could separate the classes; a positive result is not a readout or HOTA result. fit104 videos were used by earlier trials.',
        sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def load_base(arm):
    import torch
    torch.set_num_threads(2)
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from scripts.z13_default_class_trial_20261025 import load, merge
    ns = namespace(); entries = {}
    for v, h in read(FOLLOW / 'cache/report.json')['manifest'].items():
        f = FOLLOW / 'cache/videos' / (v + '.json'); assert sha(f) == h
        entries.update({int(k): x for k, x in read(f)['entries'].items()})
    d = merge(torch, load(torch, ns, arm, 'fit84', entries), load(torch, ns, arm, 'fit20', entries))
    return {k: v.numpy() for k, v in d.items()}


def chunks(folder):
    for marker in sorted(folder.glob('chunk_*.json')):
        path = marker.with_suffix('.npz'); assert sha(path) == read(marker)['sha256']
        yield path


def spatial_features(arm, d, points):
    N, K = d['mask'].shape; out = np.full((N, K, 79), np.nan, np.float32)
    index = {int(u): i for i, u in enumerate(d['uid'])}; seen = 0; max_pool_diff = max_x_diff = 0.
    for split in ('fit84', 'fit20'):
        for path in chunks(SPATIAL / arm / split):
            with np.load(path) as z:
                for r, u in enumerate(z['uid']):
                    i = index[int(u)]; seed = d['seed'][i].astype(np.float64); seen += 1
                    for k in range(z['retinal_mask'].shape[1]):
                        m = z['retinal_mask'][r, k]
                        if not m.any() or k >= K or not d['mask'][i, k]: continue
                        act = z['full'][r][m].astype(np.float64); pts = points[m]
                        max_pool_diff = max(max_pool_diff, float(np.abs(act.mean(0) - z['pooled'][r, k]).max()))
                        resp = act @ seed; c = z['centers'][r, k]; s = np.maximum(z['scales'][r, k], 1e-6)
                        rel = (pts - c) / s; w = resp - resp.min(); w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                        max_x_diff = max(max_x_diff, float(np.abs(act.mean(0) - d['x'][i, k]).max()))
                        quads = [resp[((rel[:, 0] > 0) == a) & ((rel[:, 1] > 0) == b)] for a in (False, True) for b in (False, True)]
                        top = np.sort(resp)[-max(1, len(resp) // 4):]
                        out[i, k, :11] = [resp.mean(), resp.max(), resp.std(), top.mean(), *(w @ rel), *[q.mean() if len(q) else resp.mean() for q in quads], len(resp)]
                        out[i, k, 11:] = act.std(0)
            status('spatial', arm=arm, rows_seen=seen, rows_total=N)
    assert seen == N, (arm, seen, N)
    return out, dict(pooled_vs_full=max_pool_diff, loader_x_vs_full=max_x_diff)


def temporal_features(arm, d):
    N, K = d['mask'].shape; out = np.full((N, K, 7), np.nan, np.float32); out[..., 0] = 0.
    index = {int(u): i for i, u in enumerate(d['uid'])}; found = 0
    for split in ('fit84', 'fit20'):
        for path in chunks(TEMPORAL / arm / split):
            with np.load(path) as z:
                for r, u in enumerate(z['uid']):
                    i = index.get(int(u))
                    if i is None or not z['previous_mask'][r].any(): continue
                    found += 1; prev = z['previous_x'][r][z['previous_mask'][r]].astype(np.float64)
                    seed = d['seed'][i].astype(np.float64); presp = prev @ seed
                    pn = prev / np.maximum(np.linalg.norm(prev, axis=1, keepdims=True), 1e-9)
                    for k in range(K):
                        if not d['mask'][i, k]: continue
                        x = d['x'][i, k].astype(np.float64); cos = pn @ (x / max(np.linalg.norm(x), 1e-9)); b = int(cos.argmax())
                        out[i, k] = [1., float(z['gap_frames'][r]), cos.max(), cos.mean(), presp[b], presp.max(), float(x @ seed) - presp.max()]
    return out, found


def pooled_features(d):
    x = d['x'].astype(np.float64); s = d['seed'].astype(np.float64)[:, None]
    dot = (x * s).sum(-1); cos = dot / np.maximum(np.linalg.norm(x, axis=-1) * np.linalg.norm(s, axis=-1), 1e-9)
    K = x.shape[1]
    return np.concatenate([d['x'], np.repeat(d['seed'][:, None], K, 1), dot[..., None], cos[..., None]], -1).astype(np.float32)


def evaluate(X, d, videos, folds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    known = d['kind'] >= 0; alt = d['mask'] & (d['geo'][..., -1] < 0.5) & known[:, None]
    vid = d['video']; name = np.array(videos)[vid]
    oof = np.full(d['mask'].shape, np.nan); per = {}; chosen_t = []
    for f in folds:
        def rows(vs): return np.isin(name, vs)
        tr = rows(f['train'])[:, None] & alt & (d['good'] | d['harm'])
        clf = HistGradientBoostingClassifier(**PROBE).fit(X[tr], d['good'][tr].astype(int))
        prob = np.full(d['mask'].shape, np.nan)
        part = (rows(f['calibration']) | rows(f['validation']))[:, None] & alt
        prob[part] = clf.predict_proba(X[part])[:, 1]
        def net_per_video(vs, t):
            q = rows(vs) & known; p = np.where(alt, prob, -1.); p = np.nan_to_num(p, nan=-1.)
            k = p.argmax(1); best = p[np.arange(len(p)), k]; act = q & (best >= t)
            net = np.where(act, d['good'][np.arange(len(p)), k].astype(int) - d['harm'][np.arange(len(p)), k].astype(int), 0)
            return {v: float(net[(name == v) & known].mean()) for v in vs if ((name == v) & known).any()}
        grid = [(float(np.mean(list(net_per_video(f['calibration'], t).values()))), t) for t in THRESHOLDS]
        t = max(grid)[1]; chosen_t.append(t); per.update(net_per_video(f['validation'], t))
        val = rows(f['validation'])[:, None] & alt; oof[val] = prob[val]
    lab = alt & (d['good'] | d['harm'])
    return dict(score=oof[lab], label=d['good'][lab].astype(int), video=name[np.nonzero(lab)[0]], net=per, thresholds=chosen_t)


def auc(score, label, weight=None):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score, sample_weight=weight))


def compare(results, videos, pairs):
    rng = np.random.default_rng(0); draws = rng.integers(0, len(videos), (N_BOOT, len(videos)))
    vindex = {v: i for i, v in enumerate(videos)}; out = {}
    for a, b in pairs:
        A, B = results[a], results[b]; assert np.array_equal(A['label'], B['label']) and np.array_equal(A['video'], B['video'])
        vi = np.array([vindex[v] for v in A['video']]); diffs = []
        for dr in draws:
            w = np.bincount(dr, minlength=len(videos))[vi].astype(float)
            diffs.append(auc(A['score'], A['label'], w) - auc(B['score'], B['label'], w))
        na = np.array([A['net'][v] for v in videos]); nb = np.array([B['net'][v] for v in videos]); nd = (na - nb)[draws].mean(1)
        out[f'{a}-{b}'] = dict(auc_diff=auc(A['score'], A['label']) - auc(B['score'], B['label']), auc_ci95=[float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
                               net_diff=float((na - nb).mean()), net_ci95=[float(np.quantile(nd, .025)), float(np.quantile(nd, .975))], net_better=int((na > nb).sum()), n=len(videos))
        status('bootstrap', comparison=f'{a}-{b}')
    return out


def run():
    began = time.monotonic(); p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    assert now() < DEADLINE
    videos = p['videos']; folds = p['folds']; points = np.asarray(read(GEOMETRY)['points'], np.float64)
    results = {}; checks = {}; binding = None
    for arm in ARMS:
        status('load', arm=arm); d = load_base(arm)
        if binding is None:
            binding = d['uid'].tolist(); geo0 = d['geo']; base = d
            results['geometry'] = evaluate(d['geo'], d, videos, folds); status('probe', set='geometry')
        else:
            assert binding == d['uid'].tolist()
            assert np.array_equal(d['good'], base['good']) and np.array_equal(d['harm'], base['harm']) and np.array_equal(d['mask'], base['mask'])
            checks[f'{arm}_geo_max_abs_diff_to_D'] = float(np.abs(d['geo'] - geo0).max())
        P = np.concatenate([d['geo'], pooled_features(d)], -1)
        results[arm + '_pooled'] = evaluate(P, d, videos, folds); status('probe', set=arm + '_pooled')
        S, diff = spatial_features(arm, d, points); checks[f'{arm}_spatial_alignment_max_abs_diff'] = diff
        results[arm + '_spatial'] = evaluate(np.concatenate([P, S], -1), d, videos, folds); status('probe', set=arm + '_spatial')
        T, found = temporal_features(arm, d); checks[f'{arm}_rows_with_prior'] = found
        results[arm + '_temporal'] = evaluate(np.concatenate([P, T], -1), d, videos, folds); status('probe', set=arm + '_temporal')
        del d, P, S, T
        assert time.monotonic() - began < BUDGET_SECONDS
    known = base['kind'] >= 0; alt = base['mask'] & (base['geo'][..., -1] < 0.5) & known[:, None]
    counts = dict(known_queries=int(known.sum()), helpful_alternatives=int((alt & base['good']).sum()), harmful_alternatives=int((alt & base['harm']).sum()),
                  neutral_alternatives=int((alt & ~base['good'] & ~base['harm']).sum()), queries_with_helpful=int((alt & base['good']).any(1).sum()))
    pairs = [(f'D_{s}', 'geometry') for s in SETS] + [(f'D_{s}', f'R1_{s}') for s in SETS] + [(f'D_{s}', f'C_{s}') for s in SETS] + \
            [('D_spatial', 'D_pooled'), ('D_temporal', 'D_pooled'), ('R1_pooled', 'geometry'), ('C_pooled', 'geometry')]
    paired = compare(results, videos, pairs)
    summary = {k: dict(auc=auc(r['score'], r['label']), net_mean=float(np.mean([r['net'][v] for v in videos])), thresholds=r['thresholds']) for k, r in results.items()}
    for k, r in results.items():
        np.savez_compressed(RUN / f'oof_{k}.npz', score=r['score'], label=r['label'], video=r['video'])
    decision = {}
    for s in SETS:
        g = paired[f'D_{s}-geometry']; fi = g['auc_ci95'][0] > 0 and g['net_ci95'][0] > 0
        suf = summary[f'D_{s}']['auc'] >= REFERENCE_AUC and summary[f'D_{s}']['net_mean'] >= .02
        fs = paired[f'D_{s}-R1_{s}']['auc_ci95'][0] > 0 and paired[f'D_{s}-C_{s}']['auc_ci95'][0] > 0
        decision[s] = dict(fly_information=bool(fi), sufficient_point=bool(suf), fly_specific=bool(fs))
    write(RUN / 'report.json', dict(complete=True, id=IDENT, counts=counts, checks=checks, summary=summary, paired=paired, decision=decision,
                                    net_per_video={k: r['net'] for k, r in results.items()}, seconds=time.monotonic() - began, cuda_initialized=False,
                                    hota_measured=False, new_heads=0))
    write(RUN / 'status.json', dict(state='completed', stage='done', updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
