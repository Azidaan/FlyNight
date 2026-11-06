"""Bounded paired replay: old/new point coordinates, identical saved vectors."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_bdd_video_rate_experiment import endpoint_motion_audit
from datasets.bdd100k_loader import DATA_ROOT


def main():
    video = 'b1c66a42-6f7d68ca'
    source = ROOT / 'runs/video_rate' / video
    out = ROOT / 'runs/flyvis_repairs_20261106/coordinate_only_roi'
    out.mkdir(exist_ok=False)
    paths = dict(motion=source/'motion/flyvis_full.npz',
                 metadata=source/'motion/flyvis_full.json',
                 alignment=source/'alignment.json',
                 truth=DATA_ROOT/'labels/box_track_20/val'/f'{video}.json')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = dict(video=video, scope='paired diagnostic of point-only coordinate change; not HOTA',
        selection='existing default video of run_bdd_video_rate_experiment; all labelled intervals',
        source_sha256={k: sha(p) for k,p in paths.items()},
        paths={k: str(p) for k,p in paths.items()},
        retinal_size=[391,391], flow_gain=16.00775,
        invariants=['identical saved decoded vectors and gain', 'identical truth/matches and estimator code'],
        gate='Any nonfinite output or increased mean endpoint error blocks a claim of non-regression. Even a passed one-video check cannot establish tracking benefit or general safety.',
        corrections='No fitting, data filtering, gain tuning or scientific model training permitted.')
    (out/'protocol_before_run.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    alignment = json.loads(paths['alignment'].read_text(encoding='utf-8'))
    truth = json.loads(paths['truth'].read_text(encoding='utf-8'))
    with np.load(paths['motion'], allow_pickle=False) as data:
        points = data['points'].copy()
        vectors = data['vectors'] * protocol['flow_gain']
    width, height = alignment['video']['width'], alignment['video']['height']
    # Recover retina indices from the legacy affine mapping, then apply the
    # independently specified pixel-centre inverse. Only positions change.
    centers = points.astype(np.float64) / [width,height] * 390
    corrected = ((centers + .5) / 391 * [width,height] - .5).astype(np.float32)
    results = {}
    for label, coordinates in [('legacy_v1',points),('pixel_centers_v2',corrected)]:
        started=time.monotonic()
        results[label]=endpoint_motion_audit(truth,alignment['matches'],coordinates,vectors)
        results[label]['seconds']=time.monotonic()-started
        (out/(label+'.json')).write_text(json.dumps(results[label],indent=2),encoding='utf-8')
    old=results['legacy_v1']['mean_endpoint_error_pixels']
    new=results['pixel_centers_v2']['mean_endpoint_error_pixels']
    report=dict(protocol=protocol,shape=dict(points=list(points.shape),vectors=list(vectors.shape)),
        max_point_shift_pixels=float(np.abs(corrected-points).max()),
        old_mean_endpoint_error_pixels=old,new_mean_endpoint_error_pixels=new,
        delta_mean_endpoint_error_pixels=new-old,
        non_regression_gate_passed=bool(np.isfinite(new) and new<=old),
        original_files_unchanged=all(sha(p)==protocol['source_sha256'][k] for k,p in paths.items()),
        source_script_sha256=sha(Path(__file__)),
        limitations='Retrospective one-video point replay; does not rerun renderer or neural core; no HOTA, no claim of general quality improvement.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
