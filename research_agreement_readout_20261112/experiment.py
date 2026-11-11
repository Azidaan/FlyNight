"""Exactly 26 saved-map links and five saved working cases; no tracker imports."""
from common import *

def extract(p):
    import numpy as np
    from tracking.observed_spatial_reader import ObservedSpatialReader
    from tracking.full_template_correspondence import relative_template_plan
    from tracking.hex_linear_interpolation import sample_linear
    from tracking.channel_agreement_correspondence import agreement_channels
    from tracking.z13_readout import HexGeometry
    geom = read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    reader = ObservedSpatialReader(geom['points'], geom['eye_size'], [1280, 720])
    neighbors = HexGeometry(np.array(geom['points']), geom['eye_size']).neighbors
    result = {}
    patches = {}
    max_parity = 0.
    for dataset in ['train', 'screen']:
        rows = []
        for i, row in enumerate(p[dataset]):
            status('extracting', dataset=dataset, index=i)
            plan = relative_template_plan(reader, row['reference_box'], row['reference_crop'], row['current_box'], row['current_crop'], row['offsets'])
            assert len(plan['cells']) == row['full_cells'] and len(plan['cells']) >= 3
            maps = [representations(arrays(p['snapshots'][k]), neighbors) for k in row['keys']]
            variants = {}
            for arm in ARMS:
                winner = row['old_variants'][arm]['winner']
                a = maps[0][arm][plan['cells']]
                query = plan['query'][:, winner]
                b = sample_linear(reader.mesh, maps[1][arm], query)
                # A parity witness at the saved winner; no search or refit of baseline.
                x, y = a-a.mean(0), b-b.mean(0)
                active = np.ptp(a, axis=0) != 0
                x[:, ~active] = 0
                y[:, np.ptp(b, axis=0) == 0] = 0
                den = np.linalg.norm(x, axis=0)*np.linalg.norm(y, axis=0)
                rho = np.divide((x*y).sum(0), den, out=np.zeros(34), where=den > 0)*34/active.sum()
                err = float(abs(rho-np.array(row['old_variants'][arm]['channels'])).max())
                max_parity = max(max_parity, err)
                assert err <= 1e-10, (dataset, i, arm, err)
                values = {kind: agreement_channels(a, b, mean_penalty=kind == 'agreement') for kind in KINDS}
                assert all(v is not None and np.isfinite(v).all() for v in values.values()), 'Undefined visual evidence: stop admission'
                variants[arm] = dict(winner=winner, **values)
                prefix = f'{dataset}_{i}_{arm}'
                patches[prefix + '_a'] = a
                patches[prefix + '_b'] = b
                patches[prefix + '_query'] = query
                patches[prefix + '_cells'] = plan['cells']
            rows.append(dict(**{k: row[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'label', 'geometry', 'full_cells']}, variants=variants))
        result[dataset] = rows
    verify()
    with (OUT/'patches.npz').open('xb') as f:
        np.savez_compressed(f, **patches)
    save('features.json', dict(passed=True, **result, saved_pearson_parity_max=max_parity, old_winners_reused=True, new_maps=0))
    print('EXTRACTED', max_parity, flush=True)

def solver():
    import inspect
    import tracking.channel_identity_head as h
    source = inspect.getsource(h.fit_ridge_logistic)
    old = "    raise ArithmeticError('Fixed Newton iteration limit reached')"
    replacement = """    loss, gradient, _ = objective(theta, x, y, sw)
    residual = float(np.max(np.abs(gradient)))
    if residual <= 1e-7:
        return dict(theta=theta, loss=loss, gradient_max=residual, iterations=100,
                    class_counts=counts, threshold=0., regularization=1.)
    raise ArithmeticError('Iteration limit outside registered gradient tolerance')"""
    assert source.count(old) == 1
    ns = dict(h.__dict__)
    exec(compile(source.replace(old, replacement), 'fixed_precision_solver', 'exec'), ns)
    return ns['fit_ridge_logistic']

def run_fits(p):
    import numpy as np
    f = read(OUT/'features.json')
    rows = f['train']
    screen = f['screen']
    groups = np.array([r['video'] for r in rows])
    y = np.array([r['label'] == 'own' for r in rows], int)
    G = np.array([r['geometry'] for r in rows])
    SG = np.array([r['geometry'] for r in screen])
    norms = {x['video']: x['normalization'] for x in read(TRAIN/'predictions.json')['folds'] if x['model'] == 'G7'}
    norms['FINAL'] = read(NATIVE/'deployment.json')['geometry_normalization']
    cachepath = OUT/'fit_checkpoints.json'
    cache = read(cachepath) if cachepath.exists() else {}
    fit = solver()
    scores = {a+'__'+k: np.full(26, np.nan) for a in ARMS for k in KINDS}
    screen_scores = {a+'__'+k: np.full(5, np.nan) for a in ARMS for k in KINDS}
    receipts = []
    for video in sorted(set(groups)) + ['FINAL']:
        train = np.arange(26) if video == 'FINAL' else np.flatnonzero(groups != video)
        test = np.arange(5) if video == 'FINAL' else np.flatnonzero(groups == video)
        stats = norms[video]
        g = (G-stats['mean'])/stats['scale']
        sg = (SG-stats['mean'])/stats['scale']
        for arm in ARMS:
            for kind in KINDS:
                raw = np.array([r['variants'][arm][kind] for r in rows])
                sr = np.array([r['variants'][arm][kind] for r in screen])
                mu = raw[train].mean(0)
                sd = raw[train].std(0)
                constant = np.ptp(raw[train], axis=0) == 0
                assert (~constant).any()
                rms = float(np.sqrt(np.mean(sd[~constant]**2)))
                scale = np.where(constant, 1., rms)
                X = np.c_[g, (raw-mu)/scale]
                Z = np.c_[sg, (sr-mu)/scale] if video == 'FINAL' else X
                name = arm+'__'+kind
                checkpoint = video+'|'+name
                fingerprint = hashlib.sha256(X[train].tobytes()+y[train].tobytes()).hexdigest()
                if checkpoint in cache:
                    assert cache[checkpoint]['input_sha256'] == fingerprint
                    model = cache[checkpoint]['fit']
                else:
                    model = fit(X[train], y[train])
                    cache[checkpoint] = dict(input_sha256=fingerprint, fit=cast(model))
                    temp = cachepath.with_suffix('.pending')
                    temp.write_text(json.dumps(cache, indent=2, allow_nan=False), encoding='utf8')
                    temp.replace(cachepath)
                theta = np.array(model['theta'])
                assert len(theta) == 42 and model['gradient_max'] <= 1e-7
                target = screen_scores if video == 'FINAL' else scores
                target[name][test] = Z[test]@theta[:-1]+theta[-1]
                receipts.append(dict(video=video, model=name, fit=model, normalization=stats, signal_mean=mu, signal_scale=scale,
                                     train_indices=train, test_indices=test, input_sha256=fingerprint))
        status('fitting', completed_fits=len(receipts), total_fits=126)
        print('FITS', len(receipts), '/126', flush=True)
    assert all(np.isfinite(z).all() for z in [*scores.values(), *screen_scores.values()])
    def records(rs, outputs):
        return [dict(**{k: r[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'label']}, scores={n: z[i] for n, z in outputs.items()}) for i, r in enumerate(rs)]
    verify()
    save('predictions.json', dict(passed=True, train=records(rows, scores), screen=records(screen, screen_scores), fits=receipts))

def summarize(p):
    import numpy as np
    pred = read(OUT/'predictions.json')
    rows = pred['train']
    y = np.array([r['label'] == 'own' for r in rows])
    videos = np.array([r['video'] for r in rows])
    groups = sorted(set(videos))
    def metric(z, labels=y):
        yes = np.asarray(z) > 0
        own, foreign = int((yes & labels).sum()), int((yes & ~labels).sum())
        return dict(own_accept=own, foreign_accept=foreign, balanced_accuracy=.5*(own/labels.sum()+1-foreign/(~labels).sum()))
    outputs = {n: np.array([r['scores'][n] for r in rows]) for n in rows[0]['scores']}
    base = read(SCALE/'predictions.json')['records']
    assert all(all(a[k] == b[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'label']) for a, b in zip(rows, base))
    for arm in ARMS:
        outputs[arm+'__common_baseline'] = np.array([r['scores'][arm+'__common'] for r in base])
    metrics = {n: metric(z) for n, z in outputs.items()}
    def better(a, b):
        return bool(a['own_accept'] >= b['own_accept'] and a['foreign_accept'] <= b['foreign_accept'] and
                    (a['own_accept'] > b['own_accept'] or a['foreign_accept'] < b['foreign_accept']))
    cv = {a+'__'+k: better(metrics[a+'__'+k], metrics[a+'__common_baseline']) for a in ARMS for k in KINDS}
    sc = {r['old_id']: r for r in pred['screen']}
    screen_gates = {n: bool(all(sc[i]['scores'][n] > 0 for i in [70, 1102]) and any(sc[i]['scores'][n] > 0 for i in [232, 832]) and sc[574]['scores'][n] <= 0) for n in pred['screen'][0]['scores']}
    screen_counts = {n: metric([r['scores'][n] for r in pred['screen']], np.array([r['label'] == 'own' for r in pred['screen']])) for n in screen_gates}
    pairs = [(a+'__agreement', a+'__common_baseline') for a in ARMS] + [(a+'__variance_only', a+'__common_baseline') for a in ARMS] + [(a+'__agreement', a+'__variance_only') for a in ARMS] + [('C34__agreement', a+'__agreement') for a in ARMS[1:]]
    rng = np.random.default_rng(p['uncertainty']['seed'])
    draws = {a+' minus '+b: [] for a, b in pairs}
    omitted = 0
    for _ in range(p['uncertainty']['draws']):
        selected = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([np.flatnonzero(videos == v) for v in selected])
        yy = y[idx]
        if yy.all() or not yy.any():
            omitted += 1
            continue
        ba = {n: metric(z[idx], yy)['balanced_accuracy'] for n, z in outputs.items()}
        for a, b in pairs:
            draws[a+' minus '+b].append(ba[a]-ba[b])
    intervals = {n: dict(low=float(np.quantile(v, .025)), high=float(np.quantile(v, .975)), valid=len(v)) for n, v in draws.items()}
    component = all(intervals['C34__agreement minus '+a+'__agreement']['low'] > 0 for a in ARMS[1:])
    joint = cv['C34__agreement'] and screen_gates['C34__agreement']
    verify()
    save('summary.json', dict(passed=True, metrics=metrics, cv_gates=cv, screen_metrics=screen_counts, screen_gates=screen_gates,
                             primary_joint_gate=joint, component_advantage=component, intervals95=intervals, omitted_bootstrap_draws=omitted,
                             fits=126, new_maps=0, new_core_calls=0, full_tracker_run=False, actual_IDs_evaluated=False,
                             decision='Stop this bounded package. Primary not promoted.' if not joint else 'Admission only; tracker remains prohibited in this task.',
                             scope=p['limitations']))
    print('SUMMARY', json.dumps(cast(dict(metrics=metrics, cv=cv, screen=screen_counts, joint=joint))), flush=True)

if __name__ == '__main__':
    guard()
    p = verify()
    assert read(OUT/'before_preflight.json')['passed']
    if not (OUT/'features.json').exists():
        extract(p)
    if not (OUT/'predictions.json').exists():
        run_fits(p)
    if not (OUT/'summary.json').exists():
        summarize(p)
    status('awaiting_independent_audit', primary_joint_gate=read(OUT/'summary.json')['primary_joint_gate'])
