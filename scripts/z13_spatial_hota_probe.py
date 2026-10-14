"""Bounded GPU integration probe; reads no dev annotations and measures no HOTA."""
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def model_probe():
    import torch
    import numpy as np
    from types import SimpleNamespace
    from collections import deque
    from scripts.z13_train import Episodes
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.build_fly_signatures import RenderOnlyBank
    from tracking.z13_spatial_localizer import SpatialLocalizer
    from tracking.z13_readout import run_sequence
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    data = Episodes(protocol['threshold_videos'][:1])
    label = next(r for r in data.labels if not r[8])
    result = {}
    for arm in protocol['arms']:
        estimator = make_motion_estimator('flyvis_tiled', 30., .02)
        bank = RenderOnlyBank(estimator, max_batch=4)
        loc = SpatialLocalizer(RUN / arm / 'model.pt', bank)
        hx, mask = data.batch(torch, loc.dev, [label[0]], [label[1]])
        pts = loc.geom.points[mask[0].cpu().numpy()]
        box = np.r_[pts.min(0), pts.max(0)].astype(float)
        assert np.array_equal(loc.geom.mask_in_box(box), mask[0].cpu().numpy())
        crop = (0, 0, bank.eye_size, bank.eye_size)
        loc.hist[1] = deque([(hx[0, i], box, crop) for i in range(4)], maxlen=4)
        with torch.no_grad():
            loc._new([SimpleNamespace(track_id=1)])
            state = loc._unpack(loc.eyes[1][0][None])
            last = loc.eyes[1][1][None]
            exemplar = (loc.eyes[1][2][0][None], loc.eyes[1][2][1][None])
            for step in range(4, hx.shape[1]):
                state, feat = loc.bb.step(state, last, hx[:, step]); last = hx[:, step]
            streaming = loc.ro.logits(exemplar, feat).softmax(-1)
            batch = run_sequence(torch, loc.bb, loc.ro, hx, mask)[:, -1].softmax(-1)
            difference = float((streaming - batch).abs().max())
            assert torch.allclose(streaming, batch, atol=1e-5, rtol=1e-5), (arm, difference)
        result[arm] = {'loaded': True, 'streaming_vs_training_max_difference': difference,
                       'passed': True}
        print('[MODEL]', arm, difference, flush=True)
        del loc, bank, estimator, state, feat, streaming, batch
        gc.collect(); torch.cuda.empty_cache()
    return result


def build(video, model, out):
    from scripts.build_z13_spatial_hota import main
    original = sys.argv
    try:
        sys.argv = ['build_z13_spatial_hota.py', '--video-id', video,
                    '--z13', str(model), '--output', str(out), '--max-eyes', '4']
        main()
    finally:
        sys.argv = original


def main():
    import torch
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import atomic_json
    torch.set_num_threads(2)
    probe = RUN / 'technical'
    probe.mkdir(exist_ok=True)
    if (probe / 'report.json').exists():
        raise ValueError('Technical result already exists; refusing repeated dev probe')
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    video = protocol['videos'][0]
    began = time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        models = model_probe()
        model = torch.load(RUN / 'D/model.pt', map_location='cpu', weights_only=False)
        model['tau'] = {'tau': None}
        noaction = probe / 'no_intervention_model.pt'
        torch.save(model, noaction)
        build(video, noaction, probe / 'no_intervention')
        keys_exact = ('track_id', 'start', 'end', 'cls', 'first_box', 'last_box',
                      'n_first', 'n_last', 'first_thumb', 'last_thumb', 'first_hist', 'last_hist', 'outputs')
        with (np.load(ROOT / 'runs/fly_signatures' / video / 'signatures.npz') as ref,
              np.load(probe / 'no_intervention' / video / 'signatures.npz') as actual):
            checks = {k: np.array_equal(ref[k], actual[k]) for k in keys_exact}
            assert all(checks.values()), checks
            assert all(np.allclose(ref[k], actual[k], atol=1e-3, rtol=0) for k in ('first_hexin', 'last_hexin'))
        noinfo = json.loads((probe / 'no_intervention' / video / 'signatures.json').read_text(encoding='utf-8'))
        assert noinfo['z13_events']['interventions'] == 0
        build(video, RUN / 'D/model.pt', probe / 'active_D')
        info = json.loads((probe / 'active_D' / video / 'signatures.json').read_text(encoding='utf-8'))
        report = {'created': datetime.now().astimezone().isoformat(), 'video': video,
                  'models': models, 'no_intervention_exact': checks,
                  'no_intervention_seconds': noinfo['seconds'],
                  'active_D_seconds': info['seconds'], 'active_D_events': info['z13_events'],
                  'total_probe_seconds': time.monotonic() - began,
                  'coarse_signature_gpu_hours_upper': info['seconds'] * 35 * 3 / 3600,
                  'scope': 'One-video implementation and price probe; no HOTA, no annotation reading, no tuning',
                  'passed': True}
        atomic_json(probe / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
