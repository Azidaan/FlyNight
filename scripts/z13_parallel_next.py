"""Two bounded CPU gates for utility selection and readout differentiation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_parallel_next_20261018'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(Path(path), value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(protocol):
    for rel, expected in protocol['sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Changed frozen source: ' + rel)


def status(folder, **extra):
    write(folder / 'status.json', dict(updated=datetime.now(timezone.utc).isoformat(), **extra))


def utility(folder):
    import numpy as np
    from scripts.z13_continue import coverage_table, paired, per_video_useful
    from scripts.z13_score_swap import key, selected_keys
    from tracking.z13_utility import observable_features, leave_video_out, FULL_NAMES, GEOMETRY_NAMES
    source = read(ROOT / 'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']
    assert not set(source['fit']) & set(source['predict'])
    metadata = {}
    offset = 0
    for video in source['predict']:
        with np.load(ROOT / f'runs/z13_data/{video}.npz') as data:
            for i, crop in enumerate(data['lab_crop']):
                metadata[(video, str(offset + i))] = (crop.copy(), int(data['lab_pos'][i]))
            offset += len(data['lab_kind'])
    report = dict(arms={}, paired={}, dev_evaluated=False, hota_measured=False,
                  scope='Exploratory crossfit on reused train26, no map features or core updates')
    predictions = {}
    base = ROOT / 'runs/z13_target_center_20261018'
    key_set = None
    for arm in ('D', 'C', 'R1'):
        rows = read(base / arm / 'center_rows.json')
        current_keys = {key(r) for r in rows}
        assert len(current_keys) == len(rows)
        if key_set is not None:
            assert current_keys == key_set
        key_set = current_keys
        geom, full = [], []
        for row in rows:
            crop, pos = metadata[(row['video'], str(row['label']))]
            assert pos == row['pos']
            a, b = observable_features(row, crop)
            geom.append(a); full.append(b)
        targets = [int(bool(r.get('useful'))) if r['kind'] >= 0 else -1 for r in rows]
        videos = [r['video'] for r in rows]
        variants = dict(original=rows)
        fold_records = {}
        for name, features in (('geometry', geom), ('utility', full)):
            def progress(done, total, video):
                status(folder, state='running', stage='video-held-out logistic fitting',
                       variant=arm + ':' + name, done=done, total=total, video=video)
            scores, folds = leave_video_out(features, targets, videos, progress)
            variants[name] = [dict(r, p_present=float(s)) for r, s in zip(rows, scores)]
            fold_records[name] = folds
        results = {}
        pv = {}
        for name, out in variants.items():
            results[name] = coverage_table(out)
            for q in (.01, .02, .05):
                chosen = selected_keys(out, q)
                gated = [dict(r, useful=bool(r.get('useful')) and key(r) in chosen) for r in out]
                pv[name, q] = per_video_useful(gated)
        comparisons = {f'{q:.2f}': {f'utility-{b}': paired(pv['utility', q], pv[b, q])
                                   for b in ('original', 'geometry')} for q in (.01, .02, .05)}
        report['arms'][arm] = dict(coverage=results, paired_selected_success=comparisons,
                                  video_count=len(set(videos)), row_count=len(rows))
        predictions[arm] = variants
        write(folder / f'{arm}_folds.json', fold_records)
        write(folder / f'{arm}_rows.json', variants)
        # Partial arm output is immediately recoverable.
        write(folder / f'{arm}_done.json', report['arms'][arm])
    for other in ('C', 'R1'):
        q = .01
        def pv_selected(arm):
            rows = predictions[arm]['utility']; chosen = selected_keys(rows, q)
            return per_video_useful([dict(r, useful=bool(r.get('useful')) and key(r) in chosen) for r in rows])
        report['paired']['D_utility-' + other + '_utility'] = paired(pv_selected('D'), pv_selected(other))
    report['features'] = dict(geometry=GEOMETRY_NAMES, utility=FULL_NAMES)
    primary = report['arms']['D']['paired_selected_success']['0.01']
    report['primary_pass'] = all(v['ci95'][0] > 0 for v in primary.values())
    report['decision'] = 'candidate_for_separate_validation' if report['primary_pass'] else 'limited_observables_utility_gate_not_passed'
    report['limitations'] = ('No full-map features, no calibrated risk guarantee; unknown outcomes excluded from known precision. '
                            'Failure does not diagnose the representation. Core D was not trained on train26; readout lambda fixed on train20. '
                            'Global crossfit score ordering is exploratory. No dev, HOTA or heldout75.')
    write(folder / 'report.json', report)


def differentiable(folder):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
    import numpy as np
    import torch
    torch.cuda.is_available = lambda: False
    torch.set_default_device('cpu'); torch.set_num_threads(2)
    from tracking.z13_target_center import ridge_scores, observed_center_target
    from tracking.z13_target_center_torch import ridge_scores_torch
    rng = np.random.default_rng(19)
    x = rng.normal(size=(721, 68)); xs = rng.normal(size=(2, 721, 68)); y = rng.random(721)
    output = ridge_scores_torch(torch.tensor(x), torch.tensor(y), torch.tensor(xs), [.001, .1, 1.])
    difference = float(np.max(np.abs(output.numpy() - ridge_scores(x, y, xs, [.001, .1, 1.]))))
    assert difference < 1e-10
    small = (torch.tensor(x[:9, :4], requires_grad=True), torch.tensor(y[:9], requires_grad=True),
             torch.tensor(xs[:1, :7, :4], requires_grad=True))
    checked = torch.autograd.gradcheck(lambda a, b, c: ridge_scores_torch(a, b, c, [.1]), small,
                                       eps=1e-6, atol=1e-5, rtol=1e-4)
    status(folder, state='running', stage='real D sequence and gradient', done=1, total=2)
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_readout import FlyBackbone, N_EXEMPLAR
    source = read(ROOT / 'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']
    data = Episodes(source['fit'][:1])
    lab = next(lab for lab in data.labels if lab[2] == 1 and not lab[8])
    # The first natural positive example of the first fit video, fixed before results.
    checkpoint = torch.load(ROOT / 'runs/z13_spatial_v2_20261013/D/full/model.pt', map_location='cpu', weights_only=False)
    bank, geom, bb = old.build_arm('D', torch, checkpoint)
    bb = FlyBackbone(bb.est, trainable=True)
    assert str(bb.dev) == 'cpu'
    before = {n: p.detach().clone() for n, p in bb.net.named_parameters()}
    hx, masks = data.batch(torch, 'cpu', [lab[0]], [lab[1]])
    state = bb.init(hx[:, :1]); last = hx[:, 0]; exemplar = None
    nb = torch.as_tensor(geom.neighbors[:, 1:], device='cpu')
    for i in range(1, hx.shape[1]):
        state, feature = bb.step(state, last, hx[:, i]); last = hx[:, i]
        feature = torch.cat([feature, feature[:, :, nb].mean(-1)], 1).double()
        if i == N_EXEMPLAR - 1:
            exemplar = feature[0].T
    target = torch.tensor(observed_center_target(masks[0].numpy(), geom.points, geom.spacing))
    search = feature[0].T[None]
    scores = ridge_scores_torch(exemplar, target, search, [.1])
    reference = ridge_scores(exemplar.detach().numpy(), target.numpy(), search.detach().numpy(), [.1])
    actual_diff = float(np.max(np.abs(scores.detach().numpy() - reference)))
    assert actual_diff < 1e-10
    loss = scores.square().mean()
    grads = torch.autograd.grad(loss, bb.params, allow_unused=True)
    norms = [None if g is None else float(g.norm()) for g in grads]
    assert all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    assert any(value is not None and value > 0 for value in norms)
    assert all(torch.equal(before[n], p.detach()) for n, p in bb.net.named_parameters())
    assert not torch.cuda.is_initialized()
    write(folder / 'report.json', dict(numpy_max_error=difference, gradcheck=bool(checked),
         real_D_max_error=actual_diff, core_gradient_norms=norms, weights_unchanged=True,
         test_video=source['fit'][0], episode=int(lab[0]), search_position=int(lab[1]),
         optimizer_steps=0, cuda_initialized=False, primary_pass=True,
         decision='technical_readout_gradient_ready', scientific_improvement_measured=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['utility', 'differentiable'])
    args = ap.parse_args(); folder = RUN / args.kind
    protocol = read(RUN / 'protocol.json'); verify(protocol)
    if (folder / 'report.json').exists():
        raise RuntimeError('Already completed; do not duplicate')
    folder.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
    status(folder, state='running', stage='initializing', done=0, total=0)
    try:
        {'utility': utility, 'differentiable': differentiable}[args.kind](folder)
        verify(protocol)
        status(folder, state='completed', stage='report saved', done=1, total=1)
    except BaseException as exc:
        status(folder, state='failed', stage=str(exc), done=0, total=0)
        raise


if __name__ == '__main__':
    main()
