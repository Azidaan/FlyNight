"""One resumable spatial-readout training job, or the pre-recorded technical probe."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import (RunControl, TrainingPaused, atomic_json,
                                     provenance, restore_checkpoint, save_checkpoint)
from scripts.z13_train import Episodes, DATA, backbone_groups, fit_stats, success

RUN = ROOT / 'runs/z13_spatial_v2_20261013'
CODE = ['tracking/z13_spatial.py', 'scripts/z13_spatial_train.py',
        'scripts/z13_spatial_runner.py', 'scripts/z13_threshold_diagnostic.py']


def build(arm, torch):
    from scripts.z13_train import build as original_build
    from tracking.z13_spatial import make_spatial_readout
    est, bank, geom, bb, _, dev = original_build(arm, torch)
    ro = make_spatial_readout(torch)(34, geom.points, bank.eye_size, geom.neighbors).to(dev)
    return est, bank, geom, bb, ro, dev


def identity(protocol, job):
    ident = provenance(ROOT, DATA, sorted(set(job['fit'] + job['predict'])), {'protocol': protocol, 'job': job})
    for name in CODE:
        ident['files_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return ident


def persist_torch(torch, path, state):
    import os
    temporary = path.with_name(path.name + '.next')
    with temporary.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    verified = torch.load(temporary, map_location='cpu', weights_only=False)
    if verified.get('format') != state.get('format'):
        raise RuntimeError('Export validation failed')
    os.replace(temporary, path)


def train(torch, arm, data, bb, ro, dev, folder, ident, control, steps, batch=16, micro=4):
    from tracking.z13_readout import run_sequence
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam([{'params': list(ro.parameters()), 'lr': .003}] + backbone_groups(arm, bb))
    checkpoint = folder / 'checkpoint.pt'
    usable = [i for i, row in enumerate(data.labels) if row[2] >= 0]
    if len(usable) < batch:
        raise ValueError('Insufficient training examples')
    if checkpoint.exists():
        start, history = restore_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, ident)
    else:
        fit_stats(torch, data, bb, ro, dev, np.random.default_rng(0))
        start, history = 0, []
        save_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, start, history, ident)
    control.status(state='running', stage='localization', arm=arm, step=start, total=steps)
    last_save = time.monotonic(); began = last_save
    ro.train()
    for step in range(start, steps):
        control.check()
        picked = rng.choice(usable, size=batch, replace=False)
        obj_value = 0.
        opt.zero_grad(set_to_none=True)
        for left in range(0, batch, micro):
            labels = [data.labels[i] for i in picked[left:left + micro]]
            hx, masks = data.batch(torch, dev, [r[0] for r in labels], [r[1] for r in labels])
            logits = run_sequence(torch, bb, ro, hx, masks)
            selected = logits[torch.arange(len(labels), device=dev), torch.as_tensor([r[1] for r in labels], device=dev)]
            target = torch.as_tensor([r[3] if r[2] == 1 else 721 for r in labels], device=dev)
            loss = torch.nn.functional.cross_entropy(selected, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite localization loss')
            (loss * len(labels) / batch).backward()
            obj_value += float(loss.detach()) * len(labels) / batch
        object_gradient = 0.
        if arm == 'D':
            object_gradient = float(sum(p.grad.detach().square().sum() for p in bb.params if p.grad is not None).sqrt())
        penalty = bb.penalty() if arm == 'D' else torch.zeros((), device=dev)
        if arm == 'D':
            penalty.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for group in opt.param_groups for p in group['params']):
            raise RuntimeError('Non-finite gradients')
        opt.step()
        if hasattr(bb, 'clamp'):
            bb.clamp()
        record = {'it': step + 1, 'object': obj_value, 'penalty': float(penalty.detach()),
                  'core_object_gradient': object_gradient}
        history.append(record)
        if (step + 1) % 10 == 0 or step + 1 == steps or time.monotonic() - last_save >= 60 or control.stopped():
            save_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, step + 1, history, ident)
            last_save = time.monotonic()
            elapsed = last_save - began
            control.status(step=step + 1, total=steps, checkpoint_verified=True,
                           seconds_per_step=elapsed / (step + 1 - start),
                           remaining_seconds=elapsed / (step + 1 - start) * (steps - step - 1),
                           last_loss=obj_value)
        if (step + 1) % 50 == 0 or step == start:
            print(f'[{arm}] {step + 1}/{steps} loss={obj_value:.4f}', flush=True)
        control.check()
    return history, opt, rng


def predict(torch, data, bb, ro, dev, geom, eye_size, control):
    from tracking.z13_readout import run_sequence, center_estimate, to_frame
    from tracking.z13_spatial import quality_features
    by_episode = {}
    for i, label in enumerate(data.labels):
        if not label[8]:
            by_episode.setdefault(label[0], []).append(i)
    episodes = sorted(by_episode)
    rows = []
    ro.eval()
    with torch.no_grad():
        for left in range(0, len(episodes), 4):
            control.check()
            eps = episodes[left:left + 4]
            hx, ms = data.batch(torch, dev, eps, [max(data.labels[i][1] for i in by_episode[e]) for e in eps])
            probs = run_sequence(torch, bb, ro, hx, ms).softmax(-1).cpu().numpy()
            for j, e in enumerate(eps):
                for idx in by_episode[e]:
                    label = data.labels[idx]; p = probs[j, label[1]]
                    eye = center_estimate(p[:721], geom)
                    center = np.asarray(to_frame(eye, label[7], eye_size))
                    row = {'label': idx, 'episode': int(e), 'search_position': int(label[1]), 'kind': int(label[2]),
                           'p_present': 1 - float(p[721]), 'center': center.tolist(),
                           'pred': np.asarray(label[6], float).tolist(),
                           'features': quality_features(p, geom, eye, center, label[6], label[7]).tolist()}
                    if label[2] == 1:
                        box = np.asarray(label[5]); target = np.asarray(label[4])
                        row.update(err=float(np.linalg.norm(center - target)),
                                   err_pred=float(np.linalg.norm(np.asarray(label[6]) - target)),
                                   scale=float(max(box[2] - box[0], box[3] - box[1])),
                                   col_hit=int(p[:721].argmax()) == label[3])
                    rows.append(row)
            if left % 100 == 0:
                control.status(stage='prediction', episodes_done=left + len(eps), episodes_total=len(episodes))
    return rows


def check_geometry(data, geom, eye_size):
    from tracking.z13_readout import to_eye, to_frame
    n = 0
    for label in data.labels:
        if label[2] != 1:
            continue
        x = to_eye(label[4], label[7], eye_size)
        if not np.allclose(to_frame(x, label[7], eye_size), label[4], atol=1e-4, rtol=0):
            raise ValueError('Coordinate round trip mismatch')
        if geom.nearest(x) != label[3]:
            raise ValueError('Ground truth column does not match geometry')
        n += 1
    if n == 0:
        raise ValueError('No positive labels checked')
    return n


def probe(torch, protocol, run):
    from tracking.z13_readout import run_sequence
    folder = run / 'technical'; folder.mkdir(exist_ok=True)
    report = {'arms': {}, 'scope': 'Technical execution only; no efficacy threshold selected'}
    vids = protocol['fit'][:2]
    data = Episodes(vids)
    job = {'name': 'technical', 'fit': vids, 'predict': []}
    for arm in protocol['arms']:
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        path = folder / arm; path.mkdir(exist_ok=True)
        control = RunControl(path, run / 'stop.request', resume=True)
        est, bank, geom, bb, ro, dev = build(arm, torch)
        checked = check_geometry(data, geom, bank.eye_size)
        started = time.monotonic()
        ident = identity(protocol, {**job, 'arm': arm})
        hist, opt, rng = train(torch, arm, data, bb, ro, dev, path, ident, control, protocol['technical_steps_per_arm'])
        # A checkpoint round trip must preserve the next random batch and predictions exactly.
        hx, ms = data.batch(torch, dev, [0, min(1, len(data.off) - 2)], [0, 0])
        with torch.no_grad():
            original = run_sequence(torch, bb, ro, hx, ms).detach().clone()
            changed = hx.clone(); changed[:, :4] = 1 - changed[:, :4]
            other = run_sequence(torch, bb, ro, changed, ms)
            sensitivity = float((original - other).abs().max())
        next_np = rng.integers(0, 1000000, 16)
        next_torch = torch.rand(16, device=dev)
        with torch.no_grad():
            next(ro.parameters()).add_(1)
        step, _ = restore_checkpoint(torch, path / 'checkpoint.pt', arm, bb, ro, opt, rng, ident)
        with torch.no_grad():
            restored = run_sequence(torch, bb, ro, hx, ms)
        exact = (torch.equal(original, restored) and np.array_equal(next_np, rng.integers(0, 1000000, 16))
                 and torch.equal(next_torch, torch.rand(16, device=dev)))
        r = {'geometry_labels': checked, 'steps': step, 'seconds': time.monotonic() - started,
             'template_effect_max': sensitivity, 'checkpoint_restore_exact': exact,
             'readout_parameters': sum(p.numel() for p in ro.parameters()),
             'core_object_gradient_max': max(h['core_object_gradient'] for h in hist),
             'loss_first': hist[0]['object'], 'loss_last': hist[-1]['object']}
        r['passed'] = bool(exact and sensitivity > 1e-7 and step == protocol['technical_steps_per_arm']
                           and (arm != 'D' or r['core_object_gradient_max'] > 0))
        report['arms'][arm] = r
        atomic_json(folder / 'report.json', report)
        control.status(state='completed' if r['passed'] else 'failed', stage='technical', **r)
        print(json.dumps({'arm': arm, **r}), flush=True)
        del bb, ro, est, bank, opt
        torch.cuda.empty_cache()
        if not r['passed']:
            raise RuntimeError(f'Technical gate failed: {arm}')
    report['passed'] = len({r['readout_parameters'] for r in report['arms'].values()}) == 1
    report['projected_localization_hours'] = sum(r['seconds'] / protocol['technical_steps_per_arm']
                                                * protocol['steps'] * 3 for r in report['arms'].values()) / 3600
    atomic_json(folder / 'report.json', report)
    if not report['passed']:
        raise RuntimeError('Unequal readouts')


def quality(torch, protocol, arm, run):
    from tracking.z13_spatial import make_quality_head
    from scripts.z13_threshold_diagnostic import exact_curve
    folder = run / arm; folder.mkdir(exist_ok=True)
    protocol_hash = hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest()
    folds = []
    for name in ('fold0', 'fold1'):
        p = folder / name
        meta = json.loads((p / 'identity.json').read_text(encoding='utf-8'))['config']['job']
        if set(meta['fit']) & set(meta['predict']):
            raise ValueError('Leakage in quality training')
        folds.extend(json.loads((p / 'predictions.json').read_text(encoding='utf-8')))
    labelled = [r for r in folds if r['kind'] >= 0]
    x = torch.tensor([r['features'] for r in labelled], dtype=torch.float32)
    y = torch.tensor([float(success(r)) for r in labelled], dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite quality inputs')
    torch.manual_seed(0)
    head = make_quality_head(torch)()
    head.mu.copy_(x.mean(0)); head.sd.copy_(x.std(0).clamp_min(1e-3))
    opt = torch.optim.Adam(head.parameters(), lr=protocol['quality_lr'])
    generator = torch.Generator().manual_seed(0)
    path = folder / 'quality_checkpoint.pt'
    start = 0
    if path.exists():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck['protocol_hash'] != protocol_hash:
            raise ValueError('Quality resume identity mismatch')
        head.load_state_dict(ck['head']); opt.load_state_dict(ck['optimizer'])
        generator.set_state(ck['generator']); start = ck['step']
    control = RunControl(folder / 'quality', run / 'stop.request', resume=True)
    for step in range(start, protocol['quality_steps']):
        idx = torch.randint(len(y), (protocol['quality_batch'],), generator=generator)
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[idx]), y[idx])
        if not torch.isfinite(loss):
            raise RuntimeError('Quality loss invalid')
        loss.backward(); opt.step()
        if (step + 1) % 50 == 0 or step + 1 == protocol['quality_steps'] or control.stopped():
            persist_torch(torch, path, {'format': 'z13-quality-v2', 'head': head.state_dict(), 'optimizer': opt.state_dict(),
                                       'generator': generator.get_state(), 'step': step + 1, 'protocol_hash': protocol_hash})
            control.status(state='running', stage='quality', step=step + 1, total=protocol['quality_steps'], loss=float(loss))
        control.check()
    rows = json.loads((folder / 'full/predictions.json').read_text(encoding='utf-8'))
    with torch.no_grad():
        scores = head(torch.tensor([r['features'] for r in rows])).sigmoid().numpy()
    scored = [{**r, 'presence_original': r['p_present'], 'p_present': float(score)} for r, score in zip(rows, scores)]
    rep = exact_curve(scored)
    presence = exact_curve(rows)
    atomic_json(folder / 'threshold_predictions.json', scored)
    atomic_json(folder / 'quality_curve.json', rep)
    atomic_json(folder / 'presence_curve.json', presence)
    summary = {k: v for k, v in rep.items() if k != 'curve'}
    summary['presence_only'] = {k: v for k, v in presence.items() if k != 'curve'}
    summary['quality_fit_labelled'] = len(labelled)
    summary['quality_fit_successful'] = int(y.sum())
    ck = torch.load(folder / 'full/model.pt', map_location='cpu', weights_only=False)
    ck.update(quality=head.state_dict(), tau={'tau': rep['tau']}, protocol_hash=protocol_hash)
    persist_torch(torch, folder / 'model.pt', ck)
    atomic_json(folder / 'tau.json', summary)
    control.status(state='completed', stage='model_ready', tau=rep['tau'])
    print(json.dumps({'arm': arm, **summary}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=RUN)
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--job', type=int)
    ap.add_argument('--quality', choices=['B', 'C', 'D'])
    args = ap.parse_args()
    run = args.run.resolve(); run.relative_to(ROOT)
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    import torch
    torch.set_num_threads(2)
    if args.quality:
        quality(torch, protocol, args.quality, run); return
    if args.probe:
        probe(torch, protocol, run); return
    if args.job is None:
        ap.error('Choose --probe, --job or --quality')
    job = protocol['jobs'][args.job]; arm = job['arm']
    folder = run / arm / job['name']; folder.mkdir(parents=True, exist_ok=True)
    ident = identity(protocol, job)
    if (folder / 'identity.json').exists() and json.loads((folder / 'identity.json').read_text(encoding='utf-8')) != ident:
        raise ValueError('Code/data/protocol changed')
    atomic_json(folder / 'identity.json', ident)
    control = RunControl(folder, run / 'stop.request', resume=True)
    control.check()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    est, bank, geom, bb, ro, dev = build(arm, torch)
    data = Episodes(job['fit'])
    history, _, _ = train(torch, arm, data, bb, ro, dev, folder, ident, control, protocol['steps'], protocol['batch'], protocol['micro_batch'])
    del data
    atomic_json(folder / 'loss.json', history)
    model = {'format': 'z13-spatial-v2', 'arm': arm, 'readout': ro.state_dict(), 'tau': None,
             'identity': ident, 'backbone': (bb if arm == 'B' else bb.net).state_dict()}
    persist_torch(torch, folder / 'model.pt', model)
    evaluation = Episodes(job['predict'])
    control.status(stage='prediction', step=protocol['steps'])
    rows = predict(torch, evaluation, bb, ro, dev, geom, bank.eye_size, control)
    atomic_json(folder / 'predictions.json', rows)
    control.status(state='completed', stage='predictions_saved', rows=len(rows))


if __name__ == '__main__':
    try:
        from tracking.gpu_lock import GpuLock
        with GpuLock('flyvis_fovea_gpu'):
            main()
    except TrainingPaused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
