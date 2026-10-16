"""Four matched 240-frame tracker clips in ABBA order; no production edits."""
from datetime import datetime
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_cache_clip_v1_20261015'
MODEL = ROOT / 'runs/z13_spatial_hota_v1_20261014/D/model.pt'


def main():
    from tracking.z13_checkpoint import atomic_json
    approval = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    assert approval['approved'] and datetime.now().astimezone() < datetime.fromisoformat(approval['deadline'])
    full = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
    assert json.loads((full / 'status.json').read_text(encoding='utf-8'))['state'] == 'paused'
    from tracking.z13_compiled_backend import setup
    setup()
    import cv2
    import torch
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from tracking.z13_spatial_localizer import SpatialLocalizer
    from tracking.z13_compiled_backend import enable as compile_model
    from tracking.z13_cached_params_backend import enable as cache_model
    from tracking.z13_readout import run_sequence
    from scripts.z13_train import Episodes
    from scripts.z13_spatial_hota_probe import build
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    data = Episodes(protocol['warmup_train_videos'])
    eps = list(dict.fromkeys(r[0] for r in data.labels if not r[8]))[:4]
    torch.set_num_threads(2)
    raw_capture = cv2.VideoCapture
    original = SpatialLocalizer.__init__
    report = {'passed': False, 'runs': [], 'scope': 'One fixed 240-frame excerpt, two reversed pairs. No HOTA or whole-set speed guarantee.'}

    class LimitedCapture:
        def __init__(self, *args, **kwargs):
            self.cap = raw_capture(*args, **kwargs)
            self.count = 0
        def read(self):
            if self.count >= protocol['frames']:
                return False, None
            self.count += 1
            return self.cap.read()
        def __getattr__(self, name):
            return getattr(self.cap, name)

    with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
        cv2.VideoCapture = LimitedCapture
        try:
            for index, mode in enumerate(protocol['order']):
                torch._dynamo.reset()
                def init(self, *args, **kwargs):
                    original(self, *args, **kwargs)
                    compile_model(self)
                    if mode == 'cache':
                        cache_model(self)
                    hx, mask = data.batch(torch, self.dev, eps, [0] * len(eps))
                    for batch in (1, 4):
                        run_sequence(torch, self.bb, self.ro, hx[:batch], mask[:batch])
                    torch.cuda.synchronize()
                SpatialLocalizer.__init__ = init
                destination = RUN / f'clip_{index}_{mode}'
                atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'clip',
                            'mode': mode, 'completed': index, 'total': len(protocol['order'])})
                began = time.perf_counter()
                build(protocol['video'], MODEL, destination)
                metadata = json.loads((destination / protocol['video'] / 'signatures.json').read_text(encoding='utf-8'))
                assert metadata['frames'] == protocol['frames']
                assert metadata['z13_events']['steps'] > 0
                report['runs'].append({'mode': mode, 'seconds': metadata['seconds'],
                    'outer_seconds': time.perf_counter() - began,
                    'events': metadata['z13_events'], 'source': str((destination / protocol['video'] / 'signatures.json').relative_to(ROOT))})
                atomic_json(RUN / 'report.json', report)
                gc.collect(); torch.cuda.empty_cache()
        finally:
            cv2.VideoCapture = raw_capture
            SpatialLocalizer.__init__ = original
        report['pairs'] = []
        for left, right in ((0, 1), (3, 2)):
            a, b = report['runs'][left], report['runs'][right]
            paths = [RUN / f'clip_{i}_{protocol["order"][i]}' / protocol['video'] / 'signatures.npz' for i in (left, right)]
            with np.load(paths[0]) as old, np.load(paths[1]) as new:
                assert all(np.array_equal(old[k], new[k]) for k in ('track_id', 'start', 'end', 'cls', 'frame_count'))
                assert old['outputs'].shape == new['outputs'].shape
                max_abs = float(np.abs(old['outputs'] - new['outputs']).max())
                assert np.allclose(old['outputs'], new['outputs'], atol=1e-3, rtol=0)
            assert a['events'] == b['events']
            report['pairs'].append({'saving_percent': 100 * (1 - b['seconds'] / a['seconds']),
                'output_max_abs_px': max_abs, 'same_ids_and_events': True})
        report['benefit_in_both_pairs'] = all(p['saving_percent'] > 0 for p in report['pairs'])
        report['passed'] = True
        atomic_json(RUN / 'report.json', report)
        atomic_json(RUN / 'status.json', {'state': 'completed'})
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
