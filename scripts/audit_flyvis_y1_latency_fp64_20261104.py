"""CPU-пересчёт всех входов/метрик и сверка одного полного CPU-ответа сети."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.flyvis_y1_latency_fp64_20261104 import OUT, protocol, sha, write, status
from tracking.flyvis_latency_metrics import response_metrics


def independent_metrics(flow, case):
    # Независимая реализация с явным покадровым поиском, без функции производителя.
    a = np.asarray(flow, dtype=np.float64)
    axis = case['axis']
    base = sum(float(a[t, axis]) for t in range(21, 31)) / 10
    variance = sum((float(a[t, axis]) - base) ** 2 for t in range(21, 31)) / 10
    before = sum(float(a[t, axis]) - base for t in range(45, 55)) / 10
    valid = before > max(3 * variance ** .5, 3e-5)
    first = None
    if valid:
        for t in range(55, 77):
            values = [(float(a[k, axis]) - base) / before for k in range(t, t + 3)]
            passed = all(x < 0 for x in values) if case['event'] == 'reverse' else all(abs(x) <= .2 for x in values)
            if passed:first = t - 55;break
    error = []
    for t in range(31, 79):
        target = np.zeros(2)
        if t < 55:target[axis] = case['speed']
        elif case['event'] == 'reverse':target[axis] = -case['speed']
        error.append(float(np.sqrt(sum((float(a[t, k]) - target[k]) ** 2 for k in range(2)))))
    return dict(eligible=bool(valid), latency_frames=first, epe_nominal_px=sum(error) / len(error),
                plateau_over_speed=before / case['speed'])


def main():
    import cv2
    cv2.setNumThreads(2)
    p = protocol()
    infer = json.loads((OUT / 'inference_completed.json').read_text())
    cpu = json.loads((OUT / 'cpu_reference.json').read_text())
    assert infer['complete'] and infer['frozen_parameters_unchanged'] and cpu['passed'] and not cpu['cuda_initialized']
    geometry = np.load(OUT / 'geometry.npz')
    centers = geometry['centers'];mask = geometry['mask'];pixels = geometry['pixels']
    textures = np.load(OUT / 'textures.npz')
    rows, cases = [], []
    max_eye = 0.
    t0 = time.monotonic()
    for ci, case in enumerate(p['cases']):
        status(stage='audit', done=ci, total=len(p['cases']), case=case['id'])
        input_path = OUT / 'inputs' / (case['id'] + '.npz')
        output_path = OUT / 'outputs' / (case['id'] + '.npz')
        im = json.loads(input_path.with_suffix('.json').read_text());om = json.loads(output_path.with_suffix('.json').read_text())
        assert im['sha256'] == om['input_sha256'] == sha(input_path)
        assert om['sha256'] == sha(output_path) and om['protocol_sha256'] == sha(OUT / 'protocol.json')
        inp = np.load(input_path);out = np.load(output_path)
        frames = inp['frames'];assert frames.shape == (79, 512, 512)
        assert out['flow'].shape == (2, 2, 79, 721, 2) and np.isfinite(out['flow']).all()
        gray = textures[str(case['scene'])]
        h, w = gray.shape;position = np.zeros(2, dtype=int)
        recomputed_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis_max = 0.
        previous = None
        for t, frame in enumerate(frames):
            if 31 <= t < 55:position[case['axis']] += case['speed']
            if t >= 55 and case['event'] == 'reverse':position[case['axis']] -= case['speed']
            y = (h - 512) // 2 - position[1];x = (w - 512) // 2 - position[0]
            expected = np.rint(np.clip(127.5 + case['contrast'] * (gray[y:y + 512, x:x + 512].astype(np.float32) - 127.5), 0, 255)).astype(np.uint8)
            assert np.array_equal(expected, frame)
            assert hashlib.sha256(frame.tobytes()).hexdigest() == im['frame_sha256'][t]
            resized = cv2.resize(frame, (391, 391), interpolation=cv2.INTER_AREA).astype(np.float64) / 255
            # Независимая проверка глаза интегральной суммой, не boxFilter производителя.
            padded = np.pad(resized, 6)
            integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            cy, cx = (centers + 195).T
            reference = (integral[cy + 13, cx + 13] - integral[cy, cx + 13] - integral[cy + 13, cx] + integral[cy, cx]) / 169
            diff = float(np.max(np.abs(reference - out['hexes'][t])))
            assert diff <= 1e-5, diff
            max_eye = max(max_eye, diff)
            if previous is not None:
                d = recomputed_dis.calc(previous, frame, None)[pixels[mask, 0], pixels[mask, 1]]
                dis_max = max(dis_max, float(np.max(np.abs(d - inp['dis'][t]))))
            previous = frame
        assert dis_max <= 1e-5, dis_max
        for mi, mode in enumerate(p['modes']):
            for ai, aggregation in enumerate(['mean', 'last']):
                # mean spatially only after fixed central column selection.
                trace = out['flow'][mi, ai][:, mask].astype(np.float64).mean(1)
                calculated = response_metrics(trace, case['axis'], case['speed'], case['event'])
                check = independent_metrics(trace, case)
                for key in check:
                    if isinstance(check[key], float):assert abs(check[key] - calculated[key]) <= 1e-10
                    else:assert check[key] == calculated[key]
                rows.append(dict(**case, method=mode, aggregation=aggregation, **calculated))
        dis_trace = inp['dis'].astype(np.float64).mean(1)
        dm = response_metrics(dis_trace, case['axis'], case['speed'], case['event'])
        assert independent_metrics(dis_trace, case)['latency_frames'] == dm['latency_frames']
        rows.append(dict(**case, method='DIS', aggregation='mean', **dm))
        cases.append(dict(id=case['id'], input_sha256=sha(input_path), output_sha256=sha(output_path),
                          eye_max_diff=om['eye_max_diff'], dis_max_diff=dis_max))
    write('measurements.json', dict(rows=rows, frames_per_case=79, cases=64, coordinate_system='source crop pixels, y down'))
    # Первичный вопрос: сохраняется ли лишняя задержка без интерполяции и усреднения шагов?
    pair = {}
    for r in rows:
        if r['contrast'] == 1 and r['event'] == 'reverse' and ((r['method'] == 'hold' and r['aggregation'] == 'last') or r['method'] == 'DIS'):
            pair.setdefault(r['id'], {})[r['method']] = r
    assert len(pair) == 16 and all(set(v) == {'hold', 'DIS'} for v in pair.values())
    identifiable = all(all(r['eligible'] and r['latency_frames'] is not None for r in group.values()) for group in pair.values())
    scene_deltas = []
    for scene in range(4):
        subset = [g for g in pair.values() if g['hold']['scene'] == scene]
        if all(g['hold']['eligible'] and g['DIS']['eligible'] and g['hold']['latency_frames'] is not None and g['DIS']['latency_frames'] is not None for g in subset):
            scene_deltas.append(float(np.mean([g['hold']['latency_frames'] - g['DIS']['latency_frames'] for g in subset])))
        else:scene_deltas.append(None)
    ci = None;mean = None
    if identifiable:
        x = np.asarray(scene_deltas)
        bs = x[np.random.default_rng(42).integers(0, 4, (10000, 4))].mean(1)
        ci = np.quantile(bs, [.025, .975]).tolist();mean = float(x.mean())
    advance = bool(identifiable and mean >= 1 and ci[0] > 0)
    primary = dict(identifiable=identifiable, valid_pairs=sum(all(r['eligible'] and r['latency_frames'] is not None for r in g.values()) for g in pair.values()),
                   total_pairs=16, scene_deltas_frames=scene_deltas, mean_excess_frames=mean, ci95=ci,
                   necessary_gate_passed=advance, bootstrap_unit='source image/video; four clusters',
                   limitation='Necessary observation only; not causal proof that neural time constants are responsible')
    write('analysis.json', dict(primary=primary, rows=len(rows), neural_core_modified=False,
          verdict='observable_excess_delay' if advance else ('no_registered_excess_delay' if identifiable else 'delay_not_identifiable_for_all_registered_inputs')))
    audit = dict(complete=True, passed=True, cases=cases, cases_checked=len(cases), frames_checked=64 * 79,
                 eye_max_diff=max_eye, eye_tolerance=1e-5, cpu_reference=cpu,
                 metrics_independently_recomputed=True, dis_all_frames_recomputed=True,
                 all_input_frames_independently_reconstructed=True, training=False, hota=False,
                 scope='All input frames, input projections, DIS, saved flow metrics and hashes checked; full network CPU replay only one fixed case',
                 elapsed_seconds=time.monotonic() - t0)
    write('independent_audit.json', audit)
    # Графики — измеренный временной отклик, не иллюстрация ожидаемого успеха.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    example = next(c for c in p['cases'] if c['scene'] == 0 and c['speed'] == 4 and c['axis'] == 0 and c['contrast'] == 1 and c['event'] == 'reverse')
    inp = np.load(OUT / 'inputs' / (example['id'] + '.npz'));out = np.load(OUT / 'outputs' / (example['id'] + '.npz'))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for mi, mode in enumerate(['hold', 'linear']):
        trace = out['flow'][mi, 1][:, mask, 0].astype(np.float64).mean(1)
        baseline = trace[21:31].mean();scale = (trace[45:55] - baseline).mean()
        axes[0].plot(np.arange(79) / 30, trace - baseline, label=mode)
        if abs(scale) > 3e-5:axes[1].plot(np.arange(79) / 30, (trace - baseline) / scale, label=mode)
    d = inp['dis'][:, :, 0].astype(float).mean(1)
    axes[1].plot(np.arange(79) / 30, d / 4, label='DIS / known speed', ls='--')
    for ax in axes:
        ax.axvline(55 / 30, color='black', ls=':');ax.grid(alpha=.3);ax.legend()
    axes[0].set_ylabel('Nominal displacement [px]');axes[1].set_ylabel('Response / pre-event plateau');axes[1].set_xlabel('Time [s]')
    fig.tight_layout();fig.savefig(OUT / 'response.png', dpi=160);plt.close(fig)
    status(stage='audit_completed', done=64, total=64)
    print(json.dumps(primary, ensure_ascii=False), flush=True)


if __name__ == '__main__':main()
