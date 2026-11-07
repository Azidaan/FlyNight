"""Отделить изменение состояния от повторяемости самого GPU-ядра; без HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, now


def main():
    import numpy as np, torch
    from tracking.b1_tracker_runtime import ResidualEngine
    from tracking.z13_readout import HexGeometry
    from tracking.foveated_eyes import EyeBank
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from tracking.flyvis_temporal_residual import causal_pair_activities
    from tracking.gpu_lock import GpuLock
    source = RUN / 'b1_tracker'; folder = RUN / 'b1_incremental_diagnostic'
    controller = read(source / 'controller_result.json')
    from scripts.recovery_research_service_20261027 import matching
    assert controller['all_owned_ended'] and not any(matching(k, v) for k, v in controller['owned'].items())
    torch.set_num_threads(2)
    row = read(ROOT / 'runs/b1_real_inputs_20261026/gpu_samples.json')['rows'][1]
    with np.load(ROOT / row['retinal_source']) as z:
        values = z['fr_hex'][row['prefix_start']:row['current_array_index'] + 1].astype(np.float32)
    with GpuLock('Local\\DissertationSingleGpu'):
        bank = EyeBank(make_motion_estimator('flyvis_tiled', 30., .02)); geom = HexGeometry(bank.points_local, bank.eye_size)
        hx = torch.tensor(values, device='cuda')
        results = {}
        for arm in ['C', 'D', 'R1', 'B']:
            write(folder / 'status.json', dict(state='running', arm=arm, updated=now()))
            engine = ResidualEngine(arm, bank, geom)
            outputs = {}
            weights_before = {k: v.clone() for k, v in engine.bb.net.state_dict().items()} if arm in ['C', 'D'] else {}
            with torch.no_grad():
                for mode in ['incremental1', 'direct1', 'direct2', 'incremental2']:
                    if mode.startswith('incremental'):
                        eye = engine.start([dict(hex=hx[i]) for i in range(4)])
                        for current in hx[4:]: engine.advance(eye, current)
                        ref, current = eye['reference'], eye['current']
                    else:
                        ref, current = causal_pair_activities(engine.bb, hx[None], 3)
                    outputs[mode] = (ref.clone(), current.clone())
            pairs = {}
            for a, b in [('incremental1', 'direct1'), ('direct1', 'direct2'), ('incremental1', 'incremental2')]:
                pairs[a + '-' + b] = [dict(exact=bool(torch.equal(x, y)), max_abs=float((x-y).abs().max()),
                    mean_abs=float((x-y).abs().mean()), finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
                    for x, y in zip(outputs[a], outputs[b])]
            weights_changed = [k for k, v in weights_before.items() if not torch.equal(v, engine.bb.net.state_dict()[k])]
            results[arm] = dict(pairs=pairs, weights_changed=weights_changed)
            write(folder / 'partial.json', results)
            del engine
        write(folder / 'report.json', dict(complete=True, rows=1, uid=row['uid'], results=results, checkpoint_updates=0,
            hota_measured=False, diagnosis_only=True, updated=now()))


if __name__ == '__main__':
    main()
