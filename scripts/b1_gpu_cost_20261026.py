"""Б1: ограниченный замер реального причинного прохода, не научное обучение."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/b1_real_inputs_20261026'
RUN = BASE / 'gpu_cost'
MSK = timezone(timedelta(hours=3))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare():
    baseline = read(BASE / 'report.json')
    assert baseline['passed'] and baseline['zero_decisions_exact'] and not baseline['cuda_initialized']
    assert not RUN.exists()
    RUN.mkdir()
    samples = read(BASE / 'gpu_samples.json')
    files = ['scripts/b1_gpu_cost_20261026.py', 'tracking/b1_real_inputs.py',
             'tracking/flyvis_temporal_residual.py', 'tracking/z13_readout.py',
             'scripts/z13_target_model.py', 'runs/b1_real_inputs_20261026/report.json',
             'runs/b1_real_inputs_20261026/gpu_samples.json', 'runs/b1_real_inputs_20261026/base_predictions.npz']
    files += [r['retinal_source'] for r in samples['rows']]
    start = datetime.now(MSK)
    p = dict(stage='bounded GPU cost only', created=start.isoformat(), timezone='Europe/Moscow',
             budget_seconds=120, deadline=(start + timedelta(seconds=120)).isoformat(),
             authorization='Автор поручил Б1; короткий замер стоимости после успешного соединения реальных входов. Длительное обучение не разрешается этим замером.',
             device='cuda:0', batch=1, samples=samples['rows'],
             objective='Technical sum of corrected real-region probabilities; no labels, accuracy estimate or selection by outcome.',
             steps='One head-output-only warm-up step, then three complete prefix forward/backward timings. No optimizer step on FlyVis core; no model promotion.',
             acceptance=['zero-start identity probabilities exact on real regions', 'finite nonzero gradients in free core groups after output warmup',
                         'core parameters unchanged', 'record sequence lengths, wall time and peak memory without extrapolating full training cost'],
             limitations='Single-object batches and three fixed sequence-length quantiles. Not full controller/tracker timing or training convergence. No HOTA or train26.',
             sha256={name: sha(ROOT / name) for name in sorted(set(files))})
    write('protocol.json', p)
    write('status.json', dict(state='ready', stage='GPU profile registered', updated=start.isoformat()))


def run():
    p = read(RUN / 'protocol.json')
    assert not (RUN / 'report.json').exists()
    for name, expected in p['sha256'].items():
        assert sha(ROOT / name) == expected, name
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable; no CPU substitution')
    from tracking.gpu_lock import GpuLock
    from tracking.z13_readout import FlyBackbone
    from scripts.z13_target_model import build_arm
    from tracking.flyvis_temporal_residual import TemporalResidualAssociation, causal_pair_activities
    from tracking.b1_real_inputs import corrected_identity_probability
    began = time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        _, _, frozen = build_arm('C', torch, None)
        backbone = FlyBackbone(frozen.est, trainable=True)
        assert str(backbone.dev).startswith('cuda')
        torch.manual_seed(26); torch.cuda.manual_seed_all(26)
        model = TemporalResidualAssociation().to(backbone.dev)
        before = [x.detach().clone() for x in backbone.params]
        with np.load(BASE / 'base_predictions.npz') as z:
            uid = z['uid']; probability = z['identity_probability']
        index = {int(u): i for i, u in enumerate(uid)}
        optimizer = torch.optim.SGD(model.output.parameters(), lr=.01)
        records = []
        for number, row in enumerate(p['samples']):
            write('status.json', dict(state='running', stage='GPU prefix forward/backward', sample=number,
                                     uid=row['uid'], sequence_frames=row['sequence_frames'], updated=datetime.now(MSK).isoformat()))
            with np.load(ROOT / row['retinal_source']) as z:
                hexes = torch.as_tensor(z['fr_hex'][row['prefix_start']:row['current_array_index'] + 1].astype(np.float32)[None], device=backbone.dev)
            n = len(row['choices']); columns = hexes.shape[-1]
            masks = torch.zeros((1, n, columns), dtype=torch.bool, device=backbone.dev)
            for k, ids in enumerate(row['current_masks'], start=1):
                masks[0, k, ids] = True
            reference_mask = torch.zeros((1, columns), dtype=torch.bool, device=backbone.dev)
            reference_mask[0, row['reference_mask']] = True
            base = torch.as_tensor(probability[index[row['uid']], :n][None], device=backbone.dev, dtype=torch.float64)
            valid = torch.ones((1, n), device=backbone.dev, dtype=torch.bool)
            real = valid.clone(); real[:, 0] = False
            base_logits = torch.logit(torch.nan_to_num(base, nan=.5).clamp(1e-8, 1 - 1e-8)).float()

            def forward():
                reference, current = causal_pair_activities(backbone, hexes, row['reference_index'])
                _, correction = model(base_logits, reference, current, reference_mask, masks, valid,
                                      torch.tensor([row['has_history']], device=backbone.dev))
                updated = corrected_identity_probability(base, correction.double(), real)
                return updated

            if number == 0:
                updated = forward()
                zero_exact = torch.equal(updated[real], base[real])
                assert zero_exact
                (-updated[real].mean()).backward()
                optimizer.step()  # Output layer only; the core remains unchanged.
                model.zero_grad(set_to_none=True); backbone.net.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            updated = forward()
            (-updated[real].mean()).backward()
            torch.cuda.synchronize()
            norms = [float(x.grad.norm()) if x.grad is not None else 0. for x in backbone.params]
            assert all(np.isfinite(norms)) and all(x > 0 for x in norms)
            records.append(dict(uid=row['uid'], frames=row['sequence_frames'], candidates=n - 1,
                                seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(), core_gradient_norms=norms))
            model.zero_grad(set_to_none=True); backbone.net.zero_grad(set_to_none=True)
            del updated, hexes
        unchanged = all(torch.equal(before[i], value.detach()) for i, value in enumerate(backbone.params))
        assert unchanged
        write('report.json', dict(passed=True, device=str(backbone.dev), gpu=torch.cuda.get_device_name(),
                                 batch=1, seed=26, zero_start_exact=zero_exact, samples=records,
                                 core_parameters_unchanged=unchanged, core_optimizer_steps=0, output_warmup_steps=1,
                                 total_seconds=time.monotonic() - began, hota_measured=False, train26_evaluations=0,
                                 full_training_cost_known=False, limitations=p['limitations']))
        write('status.json', dict(state='cost_profile_completed', stage='scientific pilot not started', updated=datetime.now(MSK).isoformat()))


if __name__ == '__main__':
    try:
        prepare() if sys.argv[1] == '--prepare' else run()
    except Exception as error:
        if RUN.exists():
            write('failure.json', dict(error=repr(error), updated=datetime.now(MSK).isoformat()))
        raise
