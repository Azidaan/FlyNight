"""A single authorized inference job or CPU evaluation; no model training."""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def archive_partial(paths):
    """Preserve interrupted own artifacts; never delete or reuse a corrupt cache."""
    for path in paths:
        path = path.resolve(); path.relative_to(RUN.resolve())
        if path.exists():
            destination = path.with_name(path.name + '.partial_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            assert not destination.exists()
            path.rename(destination)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['signatures', 'masa', 'eval'])
    ap.add_argument('--arm', choices=['B', 'C', 'D'])
    ap.add_argument('--video')
    args = ap.parse_args()
    os.environ['FOVEA_THREADS'] = '2'
    if args.stage == 'eval':
        from scripts.z13_spatial_hota_eval import main as evaluate
        evaluate(); return
    if not args.arm:
        ap.error('Arm required')
    if args.stage == 'signatures':
        if not args.video:
            ap.error('Video required')
        from scripts.build_z13_spatial_hota import main as build
        import numpy as np
        output = RUN / args.arm / 'signatures' / args.video
        if (output / 'signatures.npz').exists():
            try:
                metadata = json.loads((output / 'signatures.json').read_text(encoding='utf-8'))
                with np.load(output / 'signatures.npz') as saved:
                    assert int(saved['frame_count']) == metadata['frames']
                    assert len(saved['track_id']) == metadata['lives']
                    assert np.isfinite(saved['outputs']).all()
            except (OSError, ValueError, KeyError, AssertionError, EOFError):
                archive_partial([output / 'signatures.npz', output / 'signatures.json'])
        sys.argv = ['builder', '--video-id', args.video, '--z13', str(RUN / args.arm / 'model.pt'),
                    '--output', str(RUN / args.arm / 'signatures'), '--max-eyes', '4']
        build()
        import numpy as np
        output = RUN / args.arm / 'signatures' / args.video
        with (np.load(output / 'signatures.npz') as actual,
              np.load(ROOT / 'runs/fly_signatures' / args.video / 'signatures.npz') as reference):
            assert int(actual['frame_count']) == int(reference['frame_count'])
            assert np.isfinite(actual['outputs']).all()
            assert all(len(actual[k]) == len(actual['track_id']) for k in ('start', 'end', 'first_box', 'last_box'))
        return
    from tracking.gpu_lock import GpuLock
    import numpy as np
    vids = json.loads((RUN / 'videos.json').read_text(encoding='utf-8'))['videos']
    for video in vids:
        output = RUN / args.arm / 'masa' / f'{video}.npz'
        if output.exists():
            try:
                with np.load(RUN / args.arm / 'signatures' / video / 'signatures.npz') as s:
                    count = len(s['track_id'])
                with np.load(output) as e:
                    assert e['first'].shape == e['last'].shape == (count, 256)
                    assert np.isfinite(e['first']).all() and np.isfinite(e['last']).all()
            except (OSError, ValueError, KeyError, AssertionError, EOFError):
                archive_partial([output])
    with GpuLock('flyvis_fovea_gpu'):
        import torch
        torch.set_num_threads(2)
        from scripts.build_masa_embeddings import main as build
        sys.argv = ['masa', '--ids-json', str(RUN / 'videos.json'),
                    '--sig-root', str(RUN / args.arm / 'signatures'),
                    '--videos-dir', str(ROOT / 'data/bdd100k/videos/train'),
                    '--out', str(RUN / args.arm / 'masa')]
        build()
    import numpy as np
    compatibility = RUN / args.arm / 'dino_compat'
    compatibility.mkdir(exist_ok=True)
    for video in vids:
        with np.load(RUN / args.arm / 'signatures' / video / 'signatures.npz') as signatures:
            count = len(signatures['track_id'])
        with np.load(RUN / args.arm / 'masa' / f'{video}.npz') as embedding:
            assert embedding['first'].shape == embedding['last'].shape == (count, 256)
            assert np.isfinite(embedding['first']).all() and np.isfinite(embedding['last']).all()
        np.savez_compressed(compatibility / f'{video}.npz',
            first=np.zeros((count, 384), np.float16), last=np.zeros((count, 384), np.float16))


if __name__ == '__main__':
    main()
