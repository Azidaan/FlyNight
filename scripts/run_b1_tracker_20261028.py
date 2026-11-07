"""Полный замкнутый Б1: прежние веса, новые реальные назначения, без обучения."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, now

FOLDER = RUN / 'b1_tracker'


def regression(engines, folder):
    """Одна фиксированная медианная история: новый инкрементальный путь = старый."""
    import numpy as np, torch
    from tracking.flyvis_temporal_residual import causal_pair_activities
    from tracking.b1_real_inputs import corrected_identity_probability
    row = read(ROOT / 'runs/b1_real_inputs_20261026/gpu_samples.json')['rows'][1]
    with np.load(ROOT / row['retinal_source']) as z:
        hx = torch.tensor(z['fr_hex'][row['prefix_start']:row['current_array_index'] + 1].astype(np.float32), device='cuda')
    with np.load(ROOT / 'runs/b1_real_inputs_20261026/base_predictions.npz') as z:
        index = np.flatnonzero(z['uid'] == row['uid'])[0]
        probability = z['identity_probability'][index]
    from tracking.b1_learning import tensors_for_row, forward_residual
    inputs = tensors_for_row(row, hx[None], probability, 'cuda')
    masks = inputs['masks'][0].cpu().numpy()
    refmask = inputs['reference_mask'][0].cpu().numpy()
    results = {}
    for arm, engine in engines.items():
        eye = engine.start([dict(hex=hx[i]) for i in range(4)])
        for current in hx[4:]:
            engine.advance(eye, current)
        value, _ = engine.correction(eye, masks, refmask, probability)
        with torch.no_grad():
            ref, current = causal_pair_activities(engine.bb, hx[None], 3)
            assert torch.equal(ref, eye['reference']) and torch.equal(current, eye['current'])
            expected, _ = forward_residual(engine.bb, engine.head, hx[None], row, inputs, False)
        target = expected[0].cpu().numpy()
        assert np.array_equal(value, target, equal_nan=True), (arm, 'incremental residual mismatch')
        results[arm] = dict(reference_exact=True, current_exact=True, identity_probability_exact=True)
    write(folder / 'incremental_regression.json', dict(passed=True, uid=row['uid'], sequence_frames=len(hx), arms=results,
        ground_truth_used=False, checkpoint_updates=0, hota_measured=False))


def run():
    import cv2, numpy as np, torch
    from tracking.gpu_lock import GpuLock
    from tracking.b1_tracker_runtime import B1Tracker, FrozenBase, ResidualEngine, CausalMasa
    from tracking.retina import make_retina
    from tracking.z13_readout import HexGeometry
    from tracking.foveated_eyes import EyeBank
    from tracking.masa_r50 import build
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator, load_detection_cache, detections_for_frame, scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    from scripts.z13_continue import paired
    p = read(FOLDER / 'protocol.json')
    for path, h in p['sha256'].items():
        assert sha(ROOT / path) == h, path
    end = datetime.fromisoformat(read(FOLDER / 'execution_deadline.json')['deadline']).timestamp()
    began = time.monotonic()
    torch.set_num_threads(2)
    def status(**kw):
        if time.time() >= end - 15 or (FOLDER / 'stop.request').exists():
            raise TimeoutError('Original B1 tracker budget')
        write(FOLDER / 'status.json', dict(state='running', updated=now(), **kw), required=False)
    with GpuLock('Local\\DissertationSingleGpu'):
        status(stage='Загрузка прежних весов Б1 и базы')
        bank = EyeBank(make_motion_estimator('flyvis_tiled', 30., .02))
        geom = HexGeometry(bank.points_local, bank.eye_size)
        np.savez_compressed(FOLDER / 'geometry.npz', points=geom.points, eye_size=geom.eye_size)
        base = FrozenBase()
        engines = {arm: ResidualEngine(arm, bank, geom) for arm in ['C', 'D', 'R1', 'B']}
        regression(engines, FOLDER)
        encoder = build()
        buffers = {k: v.clone() for k, v in encoder.named_buffers()}
        variants = ['baseline', 'appearance', 'zero', 'C', 'D', 'R1', 'B']
        per = {a: {} for a in variants}
        proof, timing, hashes = {}, {}, {}
        for index, source in enumerate(p['sources']):
            v = source['video']; started = time.monotonic()
            assert not (FOLDER / 'completed' / (v + '.json')).exists(), 'Never repeat completed video'
            cache = load_detection_cache(ROOT / source['detections'])
            expected = read(ROOT / source['baseline'])
            alignment = read(ROOT / source['alignment'])['matches']
            anchors = {int(a['video_frame_index']): i for i, a in enumerate(alignment)}
            assert len(anchors) == len(expected)
            server = CausalMasa(encoder)
            trackers = {a: B1Tracker(SimpleNamespace(**p['config']), bank, geom, server, base,
                engines['D'] if a == 'zero' else engines.get(a), disabled=a == 'baseline', zero=a == 'zero') for a in variants}
            retina = make_retina('lamina', 30., blur_downscale=4)
            cap = cv2.VideoCapture(str(ROOT / source['video_path'])); assert cap.isOpened()
            outputs = {a: [] for a in variants}; raw = {a: [] for a in variants}
            for frame in range(int(cache['frame_count'])):
                if frame % 25 == 0:
                    status(stage='Б1 в реальном первом назначении', video=v, index=index, total=len(p['sources']),
                           frame=frame, total_frames=int(cache['frame_count']), encoded_frames=server.encoded_frames,
                           queries={a: len(t.events) for a, t in trackers.items()})
                ok, image = cap.read(); assert ok
                eye = retina.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                boxes, conf, classes = detections_for_frame(cache, frame)
                server.start(frame, image, boxes, conf)
                det = _DetSlice(boxes, conf, classes)
                frame_outputs = {}
                for a, tracker in trackers.items():
                    tracker.begin(frame, eye)
                    out = np.asarray(tracker.update(det), np.float32).reshape(-1, 8)
                    tracker.after()
                    frame_outputs[a] = out
                    if len(out):
                        raw[a].append(np.column_stack([np.full(len(out), frame), out[:, :7]]).astype(np.float32))
                    if frame in anchors:
                        ref = anchors[frame]
                        template = {k: expected[ref][k] for k in ['name', 'frameIndex']}
                        value = scalabel_frame(v, template, out)
                        if a == 'baseline':
                            assert value == expected[ref], (v, ref, 'disabled native JSON')
                        outputs[a].append(value)
                assert np.array_equal(frame_outputs['zero'], frame_outputs['appearance']), (v, frame, 'zero residual outputs')
                for left, right in zip(trackers['zero'].tracked_stracks + trackers['zero'].lost_stracks,
                                       trackers['appearance'].tracked_stracks + trackers['appearance'].lost_stracks):
                    assert left.track_id == right.track_id and np.array_equal(left.mean, right.mean) and np.array_equal(left.covariance, right.covariance)
            cap.release()
            assert all(torch.equal(buffers[k], value) for k, value in encoder.named_buffers())
            for a in variants:
                status(stage='Полная оценка Б1 и сохранение', video=v, variant=a, index=index)
                write(FOLDER / a / (v + '.json'), outputs[a])
                write(FOLDER / 'events' / a / (v + '.json'), trackers[a].events)
                array = np.concatenate(raw[a]) if raw[a] else np.zeros((0, 8), np.float32)
                array_path = FOLDER / 'raw' / a / (v + '.npz'); array_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(array_path, outputs=array)
                if a == 'baseline':
                    with np.load(ROOT / source['raw_array']) as z:
                        assert np.array_equal(array, z[source['raw_array_key']]), (v, 'native raw array')
                per[a][v] = source['baseline_metrics'] if a == 'baseline' else evaluate_hota(FOLDER / a, [v], split='val')
                for path in [FOLDER / a / (v + '.json'), FOLDER / 'events' / a / (v + '.json'), array_path]:
                    hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            proof[v] = dict(native_json_exact=True, native_raw_exact=True, zero_vs_appearance_outputs_and_state_exact=True,
                strict_past=all(all(e['reference_frame'] < e['frame'] and (e['previous_query_frame'] is None or e['previous_query_frame'] < e['frame']) for e in t.events) for t in trackers.values()),
                queries={a: len(t.events) for a, t in trackers.items()}, changed={a: outputs[a] != outputs['baseline'] for a in variants},
                encoded_frames=server.encoded_frames)
            timing[v] = time.monotonic() - started
            write(FOLDER / 'per_video.json', per)
            write(FOLDER / 'completed' / (v + '.json'), dict(proof=proof[v], seconds=timing[v], updated=now()))
            if index == 0:
                conservative = 2 * timing[v] * (len(p['sources']) - 1) + 300
                available = end - time.time()
                proceed = conservative < available
                write(FOLDER / 'price.json', dict(first_full_video_seconds=timing[v], conservative_remaining_seconds=conservative,
                                                 available_seconds=available, proceed=proceed))
                if not proceed:
                    write(FOLDER / 'cost_stop.json', dict(complete=False, scientific_verdict=None, completed_videos=[v],
                        remaining_videos=[s['video'] for s in p['sources'][1:]], updated=now()))
                    write(FOLDER / 'status.json', dict(state='stopped_on_cost', updated=now(), complete=False))
                    return
        pooled = {a: evaluate_hota(FOLDER / a, [s['video'] for s in p['sources']], split='val') for a in variants}
        contrasts = {'D-' + a: paired({v: m['HOTA'] for v, m in per['D'].items()}, {v: m['HOTA'] for v, m in per[a].items()})
                     for a in ['baseline', 'appearance', 'C', 'R1', 'B']}
        old = read(ROOT / p['strong_reference'])
        strong = old['per_video']
        contrasts['D-old_offline_masa'] = paired({v: m['HOTA'] for v, m in per['D'].items()},
            {v: strong['xywh_masa'][v]['HOTA'] for v in per['D']})
        # Формат прежнего отчёта проверяется в подготовке; никаких новых MASA-расчётов.
        write(FOLDER / 'report.json', dict(complete=True, per_video=per, pooled=pooled, paired=contrasts,
            proof=proof, timing=timing, output_sha256=hashes, seconds=time.monotonic() - began,
            main_acceptance=all(contrasts['D-' + a]['ci95'][0] > 0 for a in ['baseline', 'appearance', 'C', 'R1', 'B']), hota_measured=True,
            strong_reference=p['strong_reference'], strong_reference_format=list(strong), limitation=p['limitation']))
        write(FOLDER / 'status.json', dict(state='completed', updated=now()))


if __name__ == '__main__':
    run()
