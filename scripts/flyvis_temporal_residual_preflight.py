"""Одна техническая CPU-проба H1: нулевой старт и градиент в настоящую FlyVis.

Синтетические кандидаты проверяют только вычислительный путь. Это не обучение
набору видео и не измерение узнавания, пользы действия или HOTA.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/flyvis_temporal_residual_20261026'


def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    began = time.monotonic()
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    if (RUN / 'technical_report.json').exists():
        raise RuntimeError('Finished technical result already exists; do not repeat')
    for relative, expected in protocol['sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    write('status.json', dict(state='running', stage='load real FlyVis on CPU', updated=datetime.now(timezone.utc).isoformat()))
    import numpy as np
    import torch
    torch.cuda.is_available = lambda: False
    torch.set_default_device('cpu')
    torch.set_num_threads(2)
    torch.manual_seed(26)
    from scripts.z13_train import Episodes
    from scripts.z13_target_model import build_arm
    from tracking.z13_readout import FlyBackbone
    from tracking.flyvis_temporal_residual import TemporalResidualAssociation, causal_pair_activities
    data = Episodes([protocol['fit_video']])
    episode = next(i for i in range(len(data.off) - 1) if data.n_frames(i) >= 5 and data.mask[i].any())
    hexes, reference_mask = data.batch(torch, 'cpu', [episode], [0])
    _, geometry, frozen = build_arm('C', torch, None)
    backbone = FlyBackbone(frozen.est, trainable=True)
    assert str(backbone.dev) == 'cpu'
    before = {name: p.detach().clone() for name, p in backbone.net.named_parameters()}
    model = TemporalResidualAssociation()
    base = torch.tensor([[0., .1, -.1]])
    candidates = torch.zeros((1, 3, hexes.shape[-1]), dtype=torch.bool)
    left = geometry.points[:, 0] < np.median(geometry.points[:, 0])
    candidates[0, 1] = torch.from_numpy(left)
    candidates[0, 2] = torch.from_numpy(~left)
    valid = torch.ones((1, 3), dtype=torch.bool)
    history = torch.ones(1, dtype=torch.bool)
    target = torch.tensor([2])  # Synthetic target, no scientific labels.

    def forward():
        reference, current = causal_pair_activities(backbone, hexes, 3)
        return model(base, reference, current, reference_mask, candidates, valid, history), reference, current

    timed = time.monotonic()
    (initial, _), _, _ = forward()
    zero_exact = torch.equal(initial, base)
    torch.nn.functional.cross_entropy(initial, target).backward()
    initial_core_norms = [float(p.grad.norm()) if p.grad is not None else 0. for p in backbone.params]
    # A zero output projection blocks core gradients only on the first pass.
    # Open that projection with one head-only step; do not update the core.
    optimizer = torch.optim.SGD(model.output.parameters(), lr=.01)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    backbone.net.zero_grad(set_to_none=True)
    (scores, _), reference, current = forward()
    torch.nn.functional.cross_entropy(scores, target).backward()
    norms = [float(p.grad.norm()) if p.grad is not None else 0. for p in backbone.params]
    unavailable, correction = model(base, reference, current, reference_mask, candidates, valid, ~history)
    unchanged = all(torch.equal(before[name], p.detach()) for name, p in backbone.net.named_parameters())
    passed = (zero_exact and torch.equal(unavailable, base) and not correction.any().item()
              and unchanged and all(np.isfinite(norms)) and all(n > 0 for n in norms)
              and not torch.cuda.is_initialized())
    report = dict(passed=bool(passed), device='cpu', seed=26, fit_video=protocol['fit_video'],
                  episode=episode, input_shape=list(hexes.shape), reference_index=3,
                  candidate_source='synthetic left/right masks, technical gradient test only',
                  zero_start_exact=zero_exact, no_history_exact=torch.equal(unavailable, base),
                  initial_core_gradient_norms=initial_core_norms, core_gradient_norms=norms,
                  core_weights_unchanged=unchanged, core_optimizer_steps=0, head_optimizer_steps=1,
                  cuda_initialized=torch.cuda.is_initialized(), forward_backward_seconds=time.monotonic() - timed,
                  total_seconds=time.monotonic() - began, hota_measured=False, scientific_improvement_measured=False,
                  readiness='computational path only; real-region data join and tracker integration pending',
                  gpu_cost_measured=False)
    write('technical_report.json', report)
    write('status.json', dict(state='technical_check_completed' if passed else 'technical_check_failed',
                             stage='preparation; scientific pilot not started', updated=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
