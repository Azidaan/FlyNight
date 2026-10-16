"""Separate bounded cache benchmark on one warmed D model; never used by HOTA."""
from datetime import datetime
from pathlib import Path
import gc
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_cache_balanced_v1_20261015'
MODEL = ROOT / 'runs/z13_spatial_hota_v1_20261014/D/model.pt'


def main():
    from tracking.z13_checkpoint import atomic_json
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    approval = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    assert approval.get('approved') is True, 'Explicit short probe approval missing'
    assert datetime.now().astimezone() < datetime.fromisoformat(approval['deadline'])
    full = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
    status = json.loads((full / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'paused', 'Main GPU run must be preserved and paused'
    native = WindowsProcesses()
    last = sorted(full.glob('deadline_guard_*/result.json'), key=lambda p: p.stat().st_mtime)
    assert last and not any(native.creation(int(pid)) == stamp for pid, stamp in
        json.loads(last[-1].read_text(encoding='utf-8'))['known_owned_processes'].items())
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    from tracking.z13_compiled_backend import setup
    setup()
    import torch
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from scripts.z13_train import Episodes
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.build_fly_signatures import RenderOnlyBank
    from tracking.z13_spatial_localizer import SpatialLocalizer
    from tracking.z13_readout import run_sequence
    from tracking.z13_compiled_backend import enable as compile_model
    from tracking.z13_cached_params_backend import enable as cache_model
    torch.set_num_threads(2)
    data = Episodes(protocol['train_videos'])
    eps = list(dict.fromkeys(r[0] for r in data.labels if not r[8]))[:4]
    assert len(eps) == 4
    upto = [min(27, data.n_frames(e) - 5) for e in eps]
    report = {'created': datetime.now().astimezone().isoformat(), 'passed': False,
              'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              'blocks': {}, 'correctness': {}, 'parameter_api': {},
              'scope': 'Repeated same-model sequence timings; excludes video decoding, rendering, tracker, quality head and MASA. No end-to-end or HOTA claim.'}
    atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'load'})
    with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
        est = make_motion_estimator('flyvis_tiled', 30., .02)
        loc = SpatialLocalizer(MODEL, RenderOnlyBank(est, max_batch=4))
        compile_model(loc)
        hx, mask = data.batch(torch, loc.dev, eps, upto)
        original = loc.bb.net._param_api
        cache_model(loc)
        guarded = loc.bb.net._param_api
        cached = guarded()
        plain = lambda: cached
        modes = {'triton': original, 'triton_repeat': original,
                 'cache': guarded, 'cache_without_checks': plain}
        for batch in (1, 4):
            x, m = hx[:batch], mask[:batch]
            references = {}
            for name, prepare in modes.items():
                loc.bb.net._param_api = prepare
                for _ in range(3):
                    result = run_sequence(torch, loc.bb, loc.ro, x, m).softmax(-1)
                torch.cuda.synchronize()
                references[name] = result.cpu()
            a = references['triton']
            differences = {name: float((out - a).abs().max()) for name, out in references.items()}
            assert all(torch.allclose(out, a, atol=1e-5, rtol=1e-5) and
                       torch.equal(out.argmax(-1), a.argmax(-1)) for out in references.values())
            report['correctness'][str(batch)] = {'probability_max_abs': differences, 'same_argmax': True}
            blocks = []
            names = list(modes)
            for index in range(protocol['repeats']):
                measurements = {}
                order = names[index % 4:] + names[:index % 4]
                if index % 2:
                    order = order[::-1]
                for name in order:
                    loc.bb.net._param_api = modes[name]
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    run_sequence(torch, loc.bb, loc.ro, x, m)
                    torch.cuda.synchronize()
                    measurements[name] = time.perf_counter() - began
                blocks.append(measurements)
                report['blocks'][str(batch)] = blocks
                atomic_json(RUN / 'report.json', report)
                atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'sequence',
                            'batch': batch, 'completed': index + 1, 'total': protocol['repeats']})
            for name in ('triton', 'cache', 'cache_without_checks'):
                samples = []
                for _ in range(12):
                    torch.cuda.synchronize(); began = time.perf_counter()
                    for _ in range(20):
                        params = modes[name]()
                        _ = params.nodes.bias, params.nodes.time_const, params.edges.weight
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - began) / 20)
                report['parameter_api'][name] = samples
        loc.bb.net._param_api = guarded
        began = time.perf_counter()
        for _ in range(20000):
            guarded()
        report['guarded_return_microseconds'] = (time.perf_counter() - began) * 1e6 / 20000
        began = time.perf_counter()
        for _ in range(20000):
            plain()
        report['plain_return_microseconds'] = (time.perf_counter() - began) * 1e6 / 20000
        report['parameter_tensor_count'] = len(tuple(loc.bb.net.parameters()))
        report['summary'] = {}
        rng = np.random.default_rng(20261015)
        for batch, blocks in report['blocks'].items():
            base = np.asarray([r['triton'] for r in blocks])
            new = np.asarray([r['cache'] for r in blocks])
            resampled = rng.integers(0, len(base), size=(5000, len(base)))
            saving = 1 - new.sum() / base.sum()
            interval = np.quantile(1 - new[resampled].sum(1) / base[resampled].sum(1), [.025, .975]) * 100
            report['summary'][batch] = {'median_seconds': {name: float(np.median([r[name] for r in blocks])) for name in modes},
                'saving_percent': float(saving * 100), 'paired_block_interval_percent': interval.tolist(),
                'cache_benefit_supported_within_this_sequence': bool(interval[0] > 0)}
        report['passed'] = True
        atomic_json(RUN / 'report.json', report)
        atomic_json(RUN / 'status.json', {'state': 'completed'})
        print(json.dumps(report['summary']), flush=True)
        del loc, est, hx, mask
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
