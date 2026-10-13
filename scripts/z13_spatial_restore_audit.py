"""Separate serialization equality, RNG equality and CUDA reduction reproducibility."""
import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from scripts.z13_spatial_train import build, RUN
    from scripts.z13_train import Episodes, backbone_groups
    from tracking.z13_readout import run_sequence
    from tracking.z13_checkpoint import restore_checkpoint, atomic_json
    torch.set_num_threads(2)
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    data = Episodes(protocol['fit'][:2])
    est, bank, geom, bb, ro, dev = build('C', torch)
    checkpoint = RUN / 'technical/C/checkpoint.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    opt = torch.optim.Adam([{'params': list(ro.parameters()), 'lr': .003}] + backbone_groups('C', bb))
    rng = np.random.default_rng(0)
    hx, mask = data.batch(torch, dev, [0, 1], [0, 0])
    restore_checkpoint(torch, checkpoint, 'C', bb, ro, opt, rng, ck['identity'])
    weights_equal = all(torch.equal(v.cpu(), ck['readout'][k]) for k, v in ro.state_dict().items())
    backbone_equal = all(torch.equal(v.cpu(), ck['backbone'][k]) for k, v in bb.net.state_dict().items())
    with torch.no_grad():
        first = run_sequence(torch, bb, ro, hx, mask).clone()
        repeated = run_sequence(torch, bb, ro, hx, mask).clone()
    a = rng.integers(1000000, size=16); b = torch.rand(16, device=dev)
    restore_checkpoint(torch, checkpoint, 'C', bb, ro, opt, rng, ck['identity'])
    with torch.no_grad():
        restored = run_sequence(torch, bb, ro, hx, mask).clone()
        _ = run_sequence(torch, bb, ro, hx, mask)
    report = {'readout_exact': weights_equal, 'backbone_exact': backbone_equal,
              'numpy_rng_exact': bool(np.array_equal(a, rng.integers(1000000, size=16))),
              'torch_rng_exact': torch.equal(b, torch.rand(16, device=dev)),
              'repeat_max_abs': float((first - repeated).abs().max()),
              'restore_max_abs': float((first - restored).abs().max()),
              'repeat_allclose_1e5': torch.allclose(first, repeated, atol=1e-5, rtol=1e-5),
              'restore_allclose_1e5': torch.allclose(first, restored, atol=1e-5, rtol=1e-5)}
    atomic_json(RUN / 'technical/restore_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    from tracking.gpu_lock import GpuLock
    with GpuLock('flyvis_fovea_gpu'):
        main()
