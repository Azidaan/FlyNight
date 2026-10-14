"""One bounded, paired compiler probe; no training, tuning or HOTA."""
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
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014'
OLD = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    import torch
    import triton
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import atomic_json
    from tracking.z13_compiled_backend import enable
    from scripts.z13_train import Episodes
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.build_fly_signatures import RenderOnlyBank
    from tracking.z13_spatial_localizer import SpatialLocalizer
    torch.set_num_threads(2)
    torch._dynamo.config.suppress_errors = False
    protocol = json.loads((OLD / 'protocol.json').read_text(encoding='utf-8'))
    data = Episodes(protocol['threshold_videos'][:1])
    eps = list(dict.fromkeys(r[0] for r in data.labels if not r[8]))[:4]
    assert len(eps) == 4
    upto = [min(29, data.n_frames(e) - 5) for e in eps]
    began = time.monotonic()
    report = {'torch': torch.__version__, 'triton': triton.__version__,
              'gpu': torch.cuda.get_device_name(), 'models': {}, 'passed': False,
              'date': datetime.now().astimezone().isoformat()}

    def sequence(loc, hx, mask):
        state = loc.bb.init(hx[:, :1]); last = hx[:, 0]
        packed = []; probabilities = []
        for j in range(1, hx.shape[1]):
            state, feats = loc.bb.step(state, last, hx[:, j]); last = hx[:, j]
            packed.append(loc._pack(state).detach().cpu())
            if j == 3:
                ex = loc.ro.exemplar(feats, mask)
            if j >= 4:
                probabilities.append(loc.ro.logits(ex, feats).softmax(-1).detach().cpu())
        return torch.stack(packed), torch.stack(probabilities)

    try:
        with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
            for arm in protocol['arms']:
                atomic_json(RUN / 'status.json', {'state': 'probing', 'arm': arm, 'stage': 'sequence'})
                est = make_motion_estimator('flyvis_tiled', 30., .02)
                loc = SpatialLocalizer(OLD / arm / 'model.pt', RenderOnlyBank(est, max_batch=4))
                hx, mask = data.batch(torch, loc.dev, eps, upto)
                eager_state, eager_p = sequence(loc, hx, mask)
                state = loc.bb.init(hx[:, :1])
                torch.cuda.synchronize(); start = time.monotonic()
                for _ in range(30):
                    loc.bb.step(state, hx[:, 0], hx[:, 1])
                torch.cuda.synchronize(); eager_seconds = time.monotonic() - start
                info = enable(loc)
                start = time.monotonic()
                compiled_state, compiled_p = sequence(loc, hx, mask)
                torch.cuda.synchronize(); compilation_seconds = time.monotonic() - start
                assert torch.allclose(eager_state, compiled_state, atol=1e-4, rtol=1e-4), (arm, 'state')
                assert torch.allclose(eager_p, compiled_p, atol=1e-5, rtol=1e-5), (arm, 'probabilities')
                assert torch.equal(eager_p.argmax(-1), compiled_p.argmax(-1)), (arm, 'argmax')
                # Quality inference is unchanged; test its actual accept/reject decisions.
                from tracking.z13_spatial import quality_features
                from tracking.z13_readout import center_estimate, to_frame
                def decisions(probabilities):
                    result = []
                    for p in probabilities.numpy().reshape(-1, 722):
                        eye = center_estimate(p[:721], loc.geom)
                        crop = (0, 0, loc.bank.eye_size, loc.bank.eye_size)
                        center = np.asarray(to_frame(eye, crop, loc.bank.eye_size))
                        f = quality_features(p, loc.geom, eye, center, np.array([80., 80.]), crop)
                        result.append(float(loc.quality(torch.from_numpy(f)[None]).sigmoid()[0]) >= loc.tau)
                    return result
                assert decisions(eager_p) == decisions(compiled_p), (arm, 'quality decisions')
                torch.cuda.synchronize(); start = time.monotonic()
                for _ in range(30):
                    loc.bb.step(state, hx[:, 0], hx[:, 1])
                torch.cuda.synchronize(); compiled_seconds = time.monotonic() - start
                report['models'][arm] = {**info, 'sequence_steps': int(hx.shape[1]),
                    'state_max_abs': float((eager_state - compiled_state).abs().max()),
                    'probability_max_abs': float((eager_p - compiled_p).abs().max()),
                    'same_argmax_and_quality_decisions': True, 'eager_30_steps_seconds': eager_seconds,
                    'compiled_30_steps_seconds': compiled_seconds, 'step_speedup': eager_seconds / compiled_seconds,
                    'first_compilation_and_sequence_seconds': compilation_seconds}
                atomic_json(RUN / 'sequence_report.json', report)
                print(arm, report['models'][arm], flush=True)
                del loc, est, hx, mask, state, eager_state, compiled_state, eager_p, compiled_p
                gc.collect(); torch.cuda.empty_cache()
            atomic_json(RUN / 'status.json', {'state': 'probing', 'stage': 'paired_video_D'})
            from scripts.z13_spatial_hota_probe import build
            video = protocol['videos'][0]
            build(video, OLD / 'D/model.pt', RUN / 'technical/eager_D')
            original_init = SpatialLocalizer.__init__
            def compiled_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.compiler_info = enable(self)
            SpatialLocalizer.__init__ = compiled_init
            try:
                build(video, OLD / 'D/model.pt', RUN / 'technical/compiled_D')
                build(video, OLD / 'B/model.pt', RUN / 'technical/compiled_B')
            finally:
                SpatialLocalizer.__init__ = original_init
            comparisons = {}
            for arm, reference_root in [('D', RUN / 'technical/eager_D'), ('B', OLD / 'B/signatures')]:
                compiled_root = RUN / f'technical/compiled_{arm}'
                with np.load(reference_root / video / 'signatures.npz') as ref, np.load(compiled_root / video / 'signatures.npz') as got:
                    assert all(np.array_equal(ref[k], got[k]) for k in ('track_id', 'start', 'end', 'cls'))
                    assert ref['outputs'].shape == got['outputs'].shape
                    assert np.allclose(ref['outputs'], got['outputs'], atol=1e-3, rtol=0), arm
                    maximum = float(np.max(np.abs(ref['outputs'] - got['outputs'])))
                eager_info = json.loads((reference_root / video / 'signatures.json').read_text(encoding='utf-8'))
                compiled_info = json.loads((compiled_root / video / 'signatures.json').read_text(encoding='utf-8'))
                assert eager_info['z13_events'] == compiled_info['z13_events'], (arm, 'events')
                speedup = eager_info['seconds'] / compiled_info['seconds']
                comparisons[arm] = {'eager_seconds': eager_info['seconds'], 'compiled_seconds': compiled_info['seconds'],
                    'wall_speedup': speedup, 'max_output_abs_px': maximum, 'same_events_and_track_ids': True}
                assert speedup > 1, (arm, 'No full-fragment speed benefit', speedup)
            report.update({'paired_video': video, 'video_comparisons': comparisons,
                           'total_seconds': time.monotonic() - began, 'passed': True})
            atomic_json(RUN / 'report.json', report)
            atomic_json(RUN / 'status.json', {'state': 'technical_passed', 'report': str(RUN / 'report.json')})
    except Exception as exc:
        report['error'] = str(exc); report['total_seconds'] = time.monotonic() - began
        atomic_json(RUN / 'report.json', report)
        atomic_json(RUN / 'status.json', {'state': 'failed', 'reason': str(exc)})
        raise


if __name__ == '__main__':
    main()
