"""Isolated builder adapter: spatial Z13 localizer with the original tracker pipeline."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # A single process owns CUDA for its whole lifetime, including initialization.
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import atomic_json
    os.environ['FOVEA_THREADS'] = '2'
    os.environ.pop('FOVEA_GPU_LOCK', None)
    from scripts import build_fly_signatures as builder
    from tracking import z13_readout
    from tracking.z13_spatial_localizer import SpatialLocalizer
    args = builder.parse_args()
    if args.oracle or args.rewire != 'none' or args.history or args.connectome_json:
        raise ValueError('HOTA policy preserves the existing environment; unsupported option')
    args.skip_fly = True
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.cuda.is_available = lambda: False
    args.output = args.output.resolve(); args.output.relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    z13_readout.Localizer = SpatialLocalizer
    with GpuLock(None if args.cpu else 'flyvis_fovea_gpu'):
        for vid in args.video_id:
            status = args.output / 'worker_status.json'
            atomic_json(status, {'state': 'running', 'video': vid,
                                  'updated': datetime.now().astimezone().isoformat()})
            builder.build_video(vid, args)
            info = json.loads((args.output / vid / 'signatures.json').read_text(encoding='utf-8'))
            atomic_json(status, {'state': 'completed', 'video': vid,
                                  'seconds': info['seconds'], 'events': info.get('z13_events'),
                                  'updated': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
