"""Frozen, single-revision technical gate; all arrays retained for CPU audits.

Train: same 12 images as v1, boxes 60/90/120, speeds 1/2/4, 8 directions.
Fit a 2x2 no-intercept matrix on total displacement over frames 1..12.
Tests: prior two images (regression) and four previously unused TRAIN videos
(seed 12026). The same T1..T4 thresholds must pass on BOTH groups.
No tuning using either test group. Fail => stop; pass => stage 1 dev35.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/fly_efference_eye_v2'
VARS = ('old', 'legacy', 'new', 'ctrl')
from scripts.fly_efference_eye import Ctrl, DIRS, PATCH, eye_raw, patch, read_frame


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def check_deadline():
    if time.time() > float(os.environ.get('EFFERENCE_DEADLINE', 'inf')) - 120:
        raise TimeoutError('Authorized window ends; no further GPU steps.')


def fit_path_matrix(raw, true):
    """raw/true: independent paths x steps x xy; no sample selection by results."""
    x, y = np.asarray(raw).sum(1), np.asarray(true).sum(1)
    ct, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        raise ValueError('Calibration is rank deficient')
    return ct.T, {'rank': int(rank), 'condition': float(singular[0] / singular[-1]),
                  'mean_path_error': float(np.linalg.norm(x @ ct - y, axis=1).mean())}


def episode(est, img, size, scene_v, eye_v, *, grow=0., C=None, closed=False):
    from tracking.efference_eyes import EfferenceEyeBank
    from tracking.efference_eyes_v2 import ReferencedEfferenceEyeBank
    from tracking.foveated_eyes import EyeBank, crop_box
    from tracking.retina import make_retina
    classes = {'old': EyeBank, 'legacy': EfferenceEyeBank, 'new': ReferencedEfferenceEyeBank}
    out = {}
    for var in VARS:
        check_deadline()
        bank = classes[var](est, max_batch=8, decode_flow=True) if var in classes else None
        ctr = [Ctrl() for _ in range(8)]
        retina = make_retina('lamina', 30., blur_downscale=4)
        def frame_at(k):
            return np.concatenate([patch(img, (640., 400.), scene_v[d] * k) for d in range(8)], axis=1)
        origin = [np.array([PATCH * d + PATCH / 2, PATCH / 2]) for d in range(8)]
        positions = [o.copy() for o in origin]
        boxes = {d + 1: (*origin[d], size, size) for d in range(8)}
        f0 = retina.process(frame_at(0))
        if bank is not None:
            bank.birth(f0, boxes)
        prev_crops = {d: crop_box(boxes[d + 1], (PATCH * 8, PATCH)) for d in range(8)}
        for ctrl in ctr:
            ctrl.prev = f0
        raw = np.zeros((8, 12, 2)); predicted = np.zeros_like(raw)
        for k in range(1, 13):
            frame = retina.process(frame_at(k))
            preds = {}
            for d in range(8):
                center = positions[d] if closed else origin[d] + eye_v[d] * k
                side = size * (1 + grow) ** k
                preds[d + 1] = (*center, side, side)
            if bank is not None:
                bank.step(frame, preds)
            for d in range(8):
                box = preds[d + 1]
                crop = crop_box(box, (PATCH * 8, PATCH))
                if var == 'ctrl':
                    r = ctr[d].raw(frame, crop, box); ctr[d].prev = frame
                else:
                    r = eye_raw(bank, est, d + 1, box)
                raw[d, k - 1] = r
                if C is not None:
                    velocity = C[var] @ r
                    if var == 'old':
                        # Compensate actual integer crop origin; a pure scale
                        # change about center contributes zero translation.
                        pc = np.asarray(prev_crops[d]); nc = np.asarray(crop)
                        velocity += (nc[:2] + nc[2:] - pc[:2] - pc[2:]) / 2
                    predicted[d, k - 1] = velocity
                    if closed:
                        positions[d] += velocity
                prev_crops[d] = crop
        if not np.isfinite(raw).all() or not np.isfinite(predicted).all():
            raise ValueError('Non-finite eye output')
        out[var] = {'raw': raw, 'est': predicted}
    return out


def summarize(cases):
    path = DIRS * 24.
    result = {}
    for var in VARS:
        def vector_metrics(key):
            s = cases[key][var]['est'].sum(1)
            lengths = np.linalg.norm(s, axis=1)
            cosine = (s * path).sum(1) / np.maximum(lengths * 24., 1e-12)
            return float(cosine.mean()), float((lengths / 24.).mean())
        c2, q2 = vector_metrics('T2'); c3, q3 = vector_metrics('T3')
        result[var] = {
            'T1_drift_share': float(np.linalg.norm(cases['T1'][var]['est'].sum(1), axis=1).mean() / 24.),
            'T1s_drift_share': float(np.linalg.norm(cases['T1s'][var]['est'].sum(1), axis=1).mean() / 24.),
            'T2_cos': c2, 'T2_ratio': q2, 'T3_cos': c3, 'T3_ratio': q3,
            'T4_err_px': float(np.linalg.norm(cases['T4'][var]['est'].sum(1) - path, axis=1).mean()),
            'T4_no_corr_px': 24.}
    return result


def verdict(aggregate):
    n = aggregate['new']
    return {'T1': n['T1_drift_share'] <= .25 and n['T1s_drift_share'] <= .25,
            'T2': n['T2_cos'] >= .8 and .6 <= n['T2_ratio'] <= 1.4,
            'T3': n['T3_cos'] >= .8 and .6 <= n['T3_ratio'] <= 1.4,
            'T4': n['T4_err_px'] <= 12.}


def gate():
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'gate.json').exists():
        raise FileExistsError('Never overwrite a completed gate')
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    t0 = time.time()
    est = make_motion_estimator('flyvis_tiled', 30., .02)
    gathered = {v: [] for v in VARS}; truth = []
    for i, (vid, frame_id) in enumerate(protocol['calibration_images']):
        img = read_frame(vid, frame_id)
        for size in (60., 90., 120.):
            for speed in (1., 2., 4.):
                vals = episode(est, img, size, DIRS * speed, np.zeros((8, 2)))
                for var in VARS:
                    gathered[var].append(vals[var]['raw'])
                truth.append(np.repeat((DIRS * speed)[:, None], 12, axis=1))
        save(OUT / 'status.json', {'phase': 'calibration', 'done': i + 1, 'total': 12, 'elapsed': time.time() - t0})
        print(f'[CALIB] {i+1}/12 {time.time()-t0:.1f}s', flush=True)
    raw = {v: np.concatenate(gathered[v]) for v in VARS}; y = np.concatenate(truth)
    np.savez_compressed(OUT / 'calibration_raw.npz', true=y, **raw)
    C, fit = {}, {}
    for var in VARS:
        C[var], fit[var] = fit_path_matrix(raw[var], y)
    agreement = float(np.max(np.abs(raw['old'] - raw['new'])))
    if agreement > .01:
        raise AssertionError(f'Stationary eye changed unexpectedly: {agreement}')
    save(OUT / 'calibration.json', {'C': {v: C[v].tolist() for v in VARS}, 'fit': fit, 'old_new_raw_max_abs': agreement,
                                  'objective': 'sum of squared endpoint errors, frames 1..12', 'rows': len(y)})
    checks, arrays = {}, {}
    rotate = np.roll(DIRS, -2, axis=0)
    for group in ('regression_images', 'holdout_images'):
        checks[group] = {}
        for vid, frame_id in protocol[group]:
            img = read_frame(vid, frame_id)
            cases = {
                'T1': episode(est, img, 90., np.zeros((8,2)), DIRS * 2., C=C),
                'T1s': episode(est, img, 90., np.zeros((8,2)), DIRS * 2., grow=.01, C=C),
                'T2': episode(est, img, 90., DIRS * 2., np.zeros((8,2)), C=C),
                'T3': episode(est, img, 90., DIRS * 2., rotate * 2., C=C),
                'T4': episode(est, img, 90., DIRS * 2., np.zeros((8,2)), C=C, closed=True)}
            key = f'{vid}:{frame_id}'
            checks[group][key] = summarize(cases)
            for case, vv in cases.items():
                for var, fields in vv.items():
                    for field, array in fields.items():
                        arrays[f'{key}__{case}__{var}__{field}'] = array
            print(f'[CHECK] {group} {key} {checks[group][key]["new"]}', flush=True)
            save(OUT / 'checks_partial.json', checks)
    np.savez_compressed(OUT / 'checks_raw.npz', **arrays)
    agg = {g: {v: {k: float(np.mean([r[v][k] for r in checks[g].values()]))
                       for k in next(iter(checks[g].values()))[v]} for v in VARS} for g in checks}
    passed = {g: verdict(a) for g,a in agg.items()}
    report = {'checks': checks, 'agg': agg, 'passed': passed,
              'all_passed': all(all(a.values()) for a in passed.values()), 'seconds': time.time()-t0,
              'protocol_sha256': hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()}
    save(OUT / 'gate.json', report)
    save(OUT / 'status.json', {'phase': 'gate_complete', 'passed': report['all_passed'], 'elapsed': time.time()-t0})
    print(f'[GATE] {passed} -> {report["all_passed"]}', flush=True)
    return 0


if __name__ == '__main__':
    gate()
