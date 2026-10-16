"""Matched full-video timing of immutable-parameter caching; no quality claim."""
from datetime import datetime
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_compiled_backend import setup
setup()
RUN = ROOT / 'runs/z13_cache_speed_v1_20261014'
MODELS = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def cpu_npz_check(video):
    import numpy as np
    path = ROOT / 'runs/z13_triton_hota_v1_20261014/technical/compiled_D' / video / 'signatures.npz'
    def collect(z):
        need = {}
        for k in range(len(z['track_id'])):
            need.setdefault(int(z['start'][k]), []).append((k, 0, z['first_box'][k]))
            need.setdefault(int(z['end'][k]), []).append((k, 1, z['last_box'][k]))
        return [(frame, k, side, tuple(float(v) for v in box))
                for frame, items in need.items() for k, side, box in items]
    start = time.perf_counter()
    with np.load(path) as archive:
        old = collect(archive)
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in ('track_id', 'start', 'end', 'first_box', 'last_box')}
    new = collect(arrays)
    new_seconds = time.perf_counter() - start
    assert old == new
    return {'source': str(path.relative_to(ROOT)), 'items': len(old), 'exactly_equal': True,
            'repeated_unpack_seconds': old_seconds, 'load_once_seconds': new_seconds,
            'reduction_percent': 100 * (1 - new_seconds / old_seconds),
            'scope': 'Only CPU construction of MASA frame/box task list; not total MASA or GPU speed'}


def main():
    import torch
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import atomic_json
    from tracking.z13_compiled_backend import enable as compile_model
    from tracking.z13_cached_params_backend import enable as cache_params
    from tracking.z13_spatial_localizer import SpatialLocalizer
    torch.set_num_threads(2)
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    video = protocol['video']
    prior_path = RUN / 'report.attempt1.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {}
    report = {'created': datetime.now().astimezone().isoformat(), 'video': video,
              'passed': False, 'gpu': torch.cuda.get_device_name(),
              'sequence': {}, 'timings': {}, 'comparison': protocol['comparison']}
    atomic_json(RUN / 'report.json', report)
    try:
        atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'cpu_npz_check'})
        report['npz_cpu'] = cpu_npz_check(video)
        atomic_json(RUN / 'report.json', report)
        from scripts.z13_train import Episodes
        from scripts.run_bdd_video_rate_experiment import make_motion_estimator
        from scripts.build_fly_signatures import RenderOnlyBank
        from tracking.z13_readout import run_sequence
        from scripts.z13_spatial_hota_probe import build
        threshold = json.loads((MODELS / 'protocol.json').read_text(encoding='utf-8'))['threshold_videos']
        data = Episodes(threshold[:1])
        eps = list(dict.fromkeys(r[0] for r in data.labels if not r[8]))[:4]
        upto = [min(29, data.n_frames(e) - 5) for e in eps]
        with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
            torch._dynamo.reset()
            est = make_motion_estimator('flyvis_tiled', 30., .02)
            loc = SpatialLocalizer(MODELS / 'D/model.pt', RenderOnlyBank(est, max_batch=4))
            compile_model(loc)
            hx, mask = data.batch(torch, loc.dev, eps, upto)
            atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'sequence_D'})
            before = run_sequence(torch, loc.bb, loc.ro, hx, mask).softmax(-1).cpu()
            cache_info = cache_params(loc)
            after = run_sequence(torch, loc.bb, loc.ro, hx, mask).softmax(-1).cpu()
            assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
            assert torch.equal(before.argmax(-1), after.argmax(-1))
            report['sequence']['D'] = {**cache_info, 'probability_max_abs': float((before - after).abs().max()),
                                       'same_argmax': True, 'frames': int(hx.shape[1])}
            del loc, est, hx, mask, before, after
            gc.collect(); torch.cuda.empty_cache()
            original = SpatialLocalizer.__init__
            for arm in protocol['arms']:
                results = {}
                for mode in ('triton', 'triton_cached'):
                    torch._dynamo.reset()
                    def init(self, *args, **kwargs):
                        original(self, *args, **kwargs)
                        compile_model(self)
                        if mode == 'triton_cached':
                            cache_params(self)
                    SpatialLocalizer.__init__ = init
                    atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'video', 'arm': arm, 'mode': mode})
                    destination = RUN / arm / mode
                    saved = destination / video / 'signatures.json'
                    if saved.is_file() and mode in prior.get('timings', {}).get(arm, {}):
                        results[mode] = prior['timings'][arm][mode]
                        report['timings'][arm] = results
                        atomic_json(RUN / 'report.json', report)
                        SpatialLocalizer.__init__ = original
                        continue
                    began = time.perf_counter()
                    try:
                        build(video, MODELS / arm / 'model.pt', destination)
                    finally:
                        SpatialLocalizer.__init__ = original
                    outer = time.perf_counter() - began
                    info = json.loads((destination / video / 'signatures.json').read_text(encoding='utf-8'))
                    results[mode] = {'processing_seconds': info['seconds'], 'outer_wall_seconds': outer,
                                     'events': info['z13_events'], 'source': str((destination / video / 'signatures.json').relative_to(ROOT))}
                    report['timings'][arm] = results
                    atomic_json(RUN / 'report.json', report)
                    gc.collect(); torch.cuda.empty_cache()
                with np.load(RUN / arm / 'triton' / video / 'signatures.npz') as old, np.load(RUN / arm / 'triton_cached' / video / 'signatures.npz') as new:
                    assert all(np.array_equal(old[k], new[k]) for k in ('track_id', 'start', 'end', 'cls'))
                    assert old['outputs'].shape == new['outputs'].shape
                    difference = float(np.abs(old['outputs'] - new['outputs']).max())
                    assert np.allclose(old['outputs'], new['outputs'], atol=1e-3, rtol=0)
                assert results['triton']['events'] == results['triton_cached']['events']
                results['output_max_abs_px'] = difference
                results['same_ids_and_events'] = True
                results['reduction_percent'] = 100 * (1 - results['triton_cached']['processing_seconds'] / results['triton']['processing_seconds'])
                atomic_json(RUN / 'report.json', report)
            report['passed'] = True
            report['D_faster'] = report['timings']['D']['reduction_percent'] > 0
            report['scope_note'] = 'One matched video; B unchanged; no HOTA or multi-video speed confirmation.'
            atomic_json(RUN / 'report.json', report)
            atomic_json(RUN / 'status.json', {'state': 'completed', 'report': str(RUN / 'report.json')})
            print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        atomic_json(RUN / 'report.json', report)
        atomic_json(RUN / 'status.json', {'state': 'failed', 'reason': str(exc)})
        raise


if __name__ == '__main__':
    main()
