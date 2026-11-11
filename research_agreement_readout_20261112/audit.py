"""Independent interpolation, agreement, fit and decision checks; no refitting."""
from common import *

def main():
    guard()
    p = verify()
    import numpy as np
    from scipy.spatial import Delaunay
    from scipy.special import expit
    features = read(OUT/'features.json')
    pred = read(OUT/'predictions.json')
    summary = read(OUT/'summary.json')
    geo = read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    points = np.array(geo['points'], dtype=np.float64)
    mesh = Delaunay(points)
    # Independently reconstruct the six-neighbor averaging index, including
    # the production geometry's boundary padding and float32 coordinates.
    from tracking.z13_readout import HexGeometry
    neighbors = HexGeometry(np.array(geo['points']), geo['eye_size']).neighbors
    errors = dict(interpolation=0., agreement=0., normalization=0., score=0., gradient=0., loss=0., duplication=0.)
    checked = 0
    with np.load(OUT/'patches.npz', allow_pickle=False) as archive:
        for dataset in ['train', 'screen']:
            for i, (spec, row) in enumerate(zip(p[dataset], features[dataset])):
                assert all(spec[k] == row[k] for k in ['video', 'frame', 'old_id', 'raw_id', 'label', 'geometry', 'full_cells'])
                assert all(fr <= spec['frame'] for k in spec['keys'] for fr, crop in p['snapshots'][k]['sequence'])
                maps = [representations(arrays(p['snapshots'][k]), neighbors) for k in spec['keys']]
                old, box, rc, cc = [np.array(spec[k]) for k in ['reference_box', 'current_box', 'reference_crop', 'current_crop']]
                physical = points*(rc[2:]-rc[:2])/geo['eye_size']+rc[:2]
                trial = box[:2]+(physical-old[:2])*(box[2:]-box[:2])/(old[2:]-old[:2])
                trial = trial[:, None]+np.array(spec['offsets'])
                allq = (trial-cc[:2])/((cc[2:]-cc[:2])/geo['eye_size'])
                cells = np.flatnonzero(((physical >= old[:2]) & (physical <= old[2:])).all(1) & (mesh.find_simplex(allq) >= 0).all(1))
                assert len(cells) == spec['full_cells'] >= 3
                for arm in ARMS:
                    prefix = f'{dataset}_{i}_{arm}'
                    assert np.array_equal(cells, archive[prefix+'_cells'])
                    winner = spec['old_variants'][arm]['winner']
                    assert row['variants'][arm]['winner'] == winner
                    q = allq[cells, winner]
                    assert np.array_equal(q, archive[prefix+'_query'])
                    a = maps[0][arm][cells]
                    b = np.empty_like(a)
                    for j, point in enumerate(q):
                        indices = mesh.simplices[mesh.find_simplex(point)]
                        vertices = points[indices]
                        weights = np.linalg.solve((vertices[:2]-vertices[2]).T, point-vertices[2])
                        values = maps[1][arm][indices]
                        b[j] = values[2]+weights@(values[:2]-values[2])
                    assert np.array_equal(a, archive[prefix+'_a'])
                    errors['interpolation'] = max(errors['interpolation'], float(abs(b-archive[prefix+'_b']).max()))
                    # Check moments on the exact archived interpolated values;
                    # interpolation itself is separately checked above.
                    b = archive[prefix+'_b']
                    active_a = np.ptp(a, axis=0) != 0
                    active_b = np.ptp(b, axis=0) != 0
                    assert active_a.any() and active_b.any()
                    factor = 34/int(active_a.sum())
                    for kind in KINDS:
                        value = np.zeros(34)
                        for c in range(34):
                            if not active_a[c] or not active_b[c]:
                                continue
                            aa, bb = a[:, c], b[:, c]
                            ma, mb = float(sum(aa)/len(aa)), float(sum(bb)/len(bb))
                            va = sum((aa-ma)**2)/len(aa)
                            vb = sum((bb-mb)**2)/len(bb)
                            if kind == 'agreement':
                                den = va+vb+(ma-mb)**2
                                mse = sum((aa-bb)**2)/len(aa)
                            else:
                                den = va+vb
                                mse = sum(((aa-ma)-(bb-mb))**2)/len(aa)
                            value[c] = (1-mse/den)*factor
                        err = float(abs(value-row['variants'][arm][kind]).max())
                        errors['agreement'] = max(errors['agreement'], err)
                    checked += 1
    assert errors['interpolation'] <= 1e-9 and errors['agreement'] <= 1e-9, errors
    train = features['train']
    y = np.array([r['label'] == 'own' for r in train], int)
    G = np.array([r['geometry'] for r in train])
    for f in pred['fits']:
        arm, kind = f['model'].split('__')
        ix = np.array(f['train_indices'])
        tx = np.array(f['test_indices'])
        final = f['video'] == 'FINAL'
        if final:
            assert np.array_equal(ix, np.arange(26)) and len(tx) == 5
        else:
            assert np.array_equal(ix, np.flatnonzero([r['video'] != f['video'] for r in train]))
            assert np.array_equal(tx, np.flatnonzero([r['video'] == f['video'] for r in train]))
        mean, scale = G[ix].mean(0), G[ix].std(0)
        scale[np.ptp(G[ix], axis=0) == 0] = 1
        errors['normalization'] = max(errors['normalization'], float(abs(mean-f['normalization']['mean']).max()), float(abs(scale-f['normalization']['scale']).max()))
        raw = np.array([r['variants'][arm][kind] for r in train])
        mu, sd = raw[ix].mean(0), raw[ix].std(0)
        constant = np.ptp(raw[ix], axis=0) == 0
        ss = np.where(constant, 1., np.sqrt(np.mean(sd[~constant]**2)))
        assert np.array_equal(mu, f['signal_mean']) and np.array_equal(ss, f['signal_scale'])
        X = np.c_[(G-f['normalization']['mean'])/f['normalization']['scale'], (raw-mu)/ss]
        assert hashlib.sha256(X[ix].tobytes()+y[ix].tobytes()).hexdigest() == f['input_sha256']
        x, yy = X[ix], y[ix]
        theta = np.array(f['fit']['theta'])
        assert len(theta) == 42
        weights = len(ix)/(2*np.bincount(yy)[yy])
        z = x@theta[:-1]+theta[-1]
        loss = np.mean(weights*(np.logaddexp(0, z)-yy*z))+.5*np.dot(theta[:-1], theta[:-1])
        err = weights*(expit(z)-yy)
        gradient = np.r_[x.T@err/len(ix)+theta[:-1], err.mean()]
        errors['loss'] = max(errors['loss'], float(abs(loss-f['fit']['loss'])))
        errors['gradient'] = max(errors['gradient'], float(abs(gradient).max()))
        expanded_x = np.c_[x[:, :7], x[:, 7:]/np.sqrt(2), x[:, 7:]/np.sqrt(2)]
        expanded_w = np.r_[theta[:7], theta[7:-1]/np.sqrt(2), theta[7:-1]/np.sqrt(2)]
        errors['duplication'] = max(errors['duplication'], float(abs(expanded_x@expanded_w+theta[-1]-z).max()),
                                    float(abs(expanded_w@expanded_w-theta[:-1]@theta[:-1])))
        ds = 'screen' if final else 'train'
        target = features[ds]
        values = np.array([r['variants'][arm][kind] for r in target])
        geometry = np.array([r['geometry'] for r in target])
        Z = np.c_[(geometry-f['normalization']['mean'])/f['normalization']['scale'], (values-mu)/ss]
        z = Z[tx]@theta[:-1]+theta[-1]
        saved = np.array([pred[ds][j]['scores'][f['model']] for j in tx])
        errors['score'] = max(errors['score'], float(abs(z-saved).max()))
    assert len(pred['fits']) == 126
    assert errors['normalization'] < 1e-12 and errors['gradient'] <= 1e-7
    assert all(errors[k] < 1e-12 for k in ['loss', 'score', 'duplication'])
    for dataset, name in [('train', 'metrics'), ('screen', 'screen_metrics')]:
        for arm in ARMS:
            for kind in KINDS:
                n = arm+'__'+kind
                own = sum(r['label'] == 'own' and r['scores'][n] > 0 for r in pred[dataset])
                foreign = sum(r['label'] == 'foreign' and r['scores'][n] > 0 for r in pred[dataset])
                assert own == summary[name][n]['own_accept'] and foreign == summary[name][n]['foreign_accept']
                nown = sum(r['label'] == 'own' for r in pred[dataset])
                nforeign = len(pred[dataset])-nown
                assert abs(summary[name][n]['balanced_accuracy']-.5*(own/nown+1-foreign/nforeign)) < 1e-15
    m = summary['metrics']['C34__agreement']
    cv = m['own_accept'] >= 17 and m['foreign_accept'] <= 1 and (m['own_accept'] > 17 or m['foreign_accept'] < 1)
    s = {r['old_id']: r['scores']['C34__agreement'] for r in pred['screen']}
    screen = s[70] > 0 and s[1102] > 0 and (s[232] > 0 or s[832] > 0) and s[574] <= 0
    assert summary['primary_joint_gate'] == bool(cv and screen)
    verify()
    outputs = {f.name: sha(f) for f in [OUT/'features.json', OUT/'patches.npz', OUT/'predictions.json', OUT/'summary.json', OUT/'fit_checkpoints.json']}
    save('audit.json', dict(passed=True, checked_patch_pairs=checked, checked_fits=126, max_errors=errors,
                           primary_joint_gate=bool(cv and screen), inputs_and_sources_unchanged=True,
                           outputs=outputs, protocol_sha256=sha(OUT/'protocol.json'),
                           independence='Separate barycentric solves and squared-error identity, no model refits; common saved-map loader and neighbor indices disclosed.',
                           full_tracker_run=False, actual_IDs_evaluated=False))
    status('scientific_work_complete', primary_joint_gate=bool(cv and screen), stopped=True)
    print('INDEPENDENT_AUDIT_PASSED', errors, flush=True)

if __name__ == '__main__':
    main()
