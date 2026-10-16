"""Original separate MASA versus shared features on one complete fixed B/C/D video."""
from datetime import datetime
from pathlib import Path
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_shared_masa_probe_v1_20261015'


def main():
    from tracking.z13_checkpoint import atomic_json
    approval = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    assert approval['approved'] and datetime.now().astimezone() < datetime.fromisoformat(approval['deadline'])
    from tracking.z13_compiled_backend import setup
    setup()
    import torch
    import numpy as np
    from tracking.gpu_lock import GpuLock
    from tracking.masa_r50 import build
    from scripts import build_masa_embeddings as original
    from scripts.z13_shared_masa import extract_video
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    video = protocol['video']
    torch.set_num_threads(2)
    report = {'passed': False, 'reference': {}, 'embedding_equality': {},
              'scope': 'One full video with actual B/C/D signatures; same original model, precision and per-arm box order. No full-set HOTA claim.'}
    (RUN / 'one_video.json').write_text(json.dumps({'videos': [video]}), encoding='utf-8')
    with GpuLock('flyvis_fovea_gpu'):
        for arm, source in protocol['signature_roots'].items():
            atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'reference', 'arm': arm})
            old_args = sys.argv
            began = time.perf_counter()
            try:
                sys.argv = ['masa', '--ids-json', str(RUN / 'one_video.json'), '--sig-root', str(ROOT / source),
                    '--videos-dir', str(ROOT / 'data/bdd100k/videos/train'), '--out', str(RUN / 'reference' / arm)]
                original.main()
            finally:
                sys.argv = old_args
            report['reference'][arm] = {'seconds': time.perf_counter() - began,
                                        'output': str((RUN / 'reference' / arm / f'{video}.npz').relative_to(ROOT))}
            atomic_json(RUN / 'report.json', report)
            gc.collect(); torch.cuda.empty_cache()
        atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'shared'})
        began = time.perf_counter()
        model = build()
        def progress(frame, total):
            atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'shared', 'frame': frame, 'total': total})
        report['shared'] = extract_video(model, ROOT / 'data/bdd100k/videos/train' / f'{video}.mov',
             {arm: ROOT / source / video / 'signatures.npz' for arm, source in protocol['signature_roots'].items()},
             {arm: RUN / 'shared' / arm / f'{video}.npz' for arm in protocol['signature_roots']}, progress)
        report['shared']['including_model_load_seconds'] = time.perf_counter() - began
        for arm in protocol['signature_roots']:
            with np.load(RUN / 'reference' / arm / f'{video}.npz') as a, np.load(RUN / 'shared' / arm / f'{video}.npz') as b:
                equal = all(np.array_equal(a[k], b[k]) for k in ('first', 'last'))
                report['embedding_equality'][arm] = {'bitwise_arrays_equal': equal,
                    'max_abs': max(float(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32)).max()) for k in ('first', 'last'))}
                assert equal, 'Embedding mismatch for ' + arm
        report['reference_total_seconds'] = sum(r['seconds'] for r in report['reference'].values())
        report['reduction_percent'] = 100 * (1 - report['shared']['including_model_load_seconds'] / report['reference_total_seconds'])
        report['embedding_gate_passed'] = True
        atomic_json(RUN / 'report.json', report)
        del model
        gc.collect(); torch.cuda.empty_cache()
    # The pair-feature check is CPU only, after all timing measurements.
    atomic_json(RUN / 'status.json', {'state': 'running', 'stage': 'pair_features_cpu'})
    from scripts import stitch_eval_v2 as stitch
    report['pair_feature_equality'] = {}
    for arm, source in protocol['signature_roots'].items():
        with np.load(ROOT / source / video / 'signatures.npz') as s:
            count = len(s['track_id'])
        dino = RUN / 'dino' / arm; dino.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dino / f'{video}.npz', first=np.zeros((count, 384), np.float16), last=np.zeros((count, 384), np.float16))
        task = (video, str(ROOT / source), str(ROOT / 'runs/train_cache'), 'train')
        stitch._set_dirs(150, 5., str(dino), str(RUN / 'reference' / arm))
        a = stitch.video_pairs(task)
        stitch._set_dirs(150, 5., str(dino), str(RUN / 'shared' / arm))
        b = stitch.video_pairs(task)
        equal = np.array_equal(a['pairs'], b['pairs']) and np.array_equal(a['X'], b['X'])
        report['pair_feature_equality'][arm] = {'pairs_equal': bool(np.array_equal(a['pairs'], b['pairs'])),
                                            'all_features_equal': bool(np.array_equal(a['X'], b['X'])), 'candidate_pairs': len(a['pairs'])}
        assert equal, 'Pair feature mismatch for ' + arm
        del a, b
        gc.collect()
    report['passed'] = report['reduction_percent'] > 0
    report['fixed_stitcher_input_gate_passed'] = True
    report['stitch_consequence'] = 'Exact same candidate pairs and all features imply same scores and stitching for the unchanged fixed head; full HOTA not recomputed in this probe.'
    atomic_json(RUN / 'report.json', report)
    atomic_json(RUN / 'status.json', {'state': 'completed'})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
