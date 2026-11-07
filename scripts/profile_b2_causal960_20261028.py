"""Полный причинный проход одного видео: 960/fused/native-cache/нулевой контроль.

Без optimizer.step, без новых весов и без оценки качества на train.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
from datetime import datetime
import time, sys, copy, faulthandler
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continuation_service_20261028 import RUN, service
FOLDER = RUN / 'b2_causal960_profile'


def run():
    p = service.read(FOLDER / 'protocol.json')
    for f, h in p['sha256'].items(): assert service.sha(ROOT / f) == h, f
    prior = service.read(RUN / 'b2_manifest_checks_recovery1/controller_result.json')
    assert prior['exit_code'] == 0 and prior['all_owned_ended']
    assert not any(service.matching(k, v) for k, v in prior['owned'].items())
    end = datetime.fromisoformat(service.read(FOLDER / 'execution_deadline.json')['deadline']).timestamp()
    import numpy as np
    import cv2, torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    from tracking.b2_causal_detector import FeatureRing, CausalP3Branch, detection_targets, CrowdAwareBCE
    from tracking.b2_feature_temporal import detector_features, detection_training_predictions
    from tracking.z13_readout import make_modules
    from scripts.z13_target_model import build_arm
    from tracking.gpu_lock import GpuLock
    faulthandler.dump_traceback_later(180, exit=True)

    def status(stage, **kw):
        if time.time() > end - 20: raise TimeoutError('B2 registered profile deadline')
        faulthandler.cancel_dump_traceback_later(); faulthandler.dump_traceback_later(180, exit=True)
        service.write(FOLDER / 'status.json', dict(stage=stage, updated=service.now(), **kw))

    with GpuLock('Local\\DissertationSingleGpu'):
        started = time.monotonic()
        manifest = service.read(ROOT / p['manifest'])
        cache_path = ROOT / p['cache']
        cache = np.load(cache_path)
        metadata = service.read(cache_path.with_suffix('.json'))
        assert metadata['identity']['imgsz'] == 960 and metadata['identity']['conf'] == .01
        assert metadata['identity']['weights_sha256'] == service.sha(ROOT / 'weights/baseline_seed42/best.pt')
        capture = cv2.VideoCapture(str(ROOT / manifest['video_path']))
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read(); assert ok
        yolo = YOLO(str(ROOT / 'weights/baseline_seed42/best.pt'))
        for _ in range(2):
            yolo.predict(frame, imgsz=960, conf=.01, iou=.7, device='0', verbose=False)
        predictor = yolo.predictor
        detector = predictor.model.model
        detector.eval().float().requires_grad_(False)
        detector.args = get_cfg(overrides=detector.args)
        before = {n: x.detach().clone() for n, x in detector.named_parameters()}
        head = detector.model[-1]
        _, geom, stock = build_arm('C', torch, None)
        _, Ordinary = make_modules(torch)
        torch.manual_seed(26); torch.cuda.manual_seed_all(26)
        ordinary = Ordinary(geom.neighbors).cuda()
        ordinary.dev = torch.device('cuda')
        torch.manual_seed(26)
        branch = CausalP3Branch(128, geom.points, geom.eye_size).cuda()
        ring = FeatureRing(5)
        frames = sorted({int(x) for clip in manifest['clips'] for x in clip['input_frames']})
        targets = {int(c['input_frames'][-1]): c for c in manifest['clips']}
        query = set(frames); raw_index = 0; queries = 0; clips = 0; peak_ring = 0
        last = None
        while True:
            if raw_index in query:
                status('complete video/native cache and zero full detections', frame=raw_index,
                       queries=queries, total_queries=len(query), clips=clips)
                image = predictor.preprocess([frame])
                current = detector_features(detector, image)
                assert current[0].shape[1] == 128
                with torch.no_grad():
                    native = head(current)
                    prediction = detection_training_predictions(head, current)
                    ring.append(raw_index, current)
                    sequence, observed_frames = ring.sequence()
                    if raw_index in targets:
                        assert observed_frames == targets[raw_index]['input_frames']
                        modified, correction = branch(sequence, ordinary, observed_frames, fps)
                        assert torch.count_nonzero(correction) == 0
                        zero_prediction = detection_training_predictions(head, [modified, *current[1:]])
                        assert all(torch.equal(prediction[k], zero_prediction[k]) for k in ('boxes', 'scores'))
                        zero = head([modified, *current[1:]])
                        assert torch.equal(native[0], zero[0])
                        clips += 1
                    result = predictor.postprocess(native, image, [frame])[0]
                got = result.boxes
                mask = cache['frame_idx'] == raw_index
                assert np.array_equal(got.xyxy.cpu().numpy().astype(np.float32), cache['xyxy'][mask]), ('cached boxes', raw_index)
                assert np.array_equal(got.conf.cpu().numpy().astype(np.float32), cache['conf'][mask]), ('cached scores', raw_index)
                assert np.array_equal(got.cls.cpu().numpy().astype(np.int16), cache['cls'][mask]), ('cached classes', raw_index)
                peak_ring = max(peak_ring, ring.bytes)
                queries += 1
                last = (current, image, frame.shape, sequence, observed_frames, targets.get(raw_index))
            raw_index += 1
            ok, frame = capture.read()
            if not ok: break
        capture.release()
        assert queries == len(query) and clips == len(manifest['clips'])
        full_video_seconds = time.monotonic() - started
        service.write(FOLDER / 'video_completed.json', dict(complete=True, raw_frames=raw_index,
                      query_frames=queries, clips=clips, fps=fps, native_cache_exact=True,
                      zero_raw_exact=True, peak_ring_bytes=peak_ring, seconds=full_video_seconds,
                      manifest_sha256=service.sha(ROOT / p['manifest']), updated=service.now()))
        status('loss/strong separately trainable controls; no optimizer updates')
        current, image, original_shape, sequence, observed_frames, clip = last
        assert clip is not None
        truth = service.read(ROOT / manifest['gt_only_detection_targets'])
        entry = truth[clip['target_reference_index']]
        assert entry['name'] == clip['target_reference_name']
        target = detection_targets(entry, detector.names, original_shape, image.shape)
        batch = dict(img=image, batch_idx=torch.zeros(len(target['cls']), device='cuda', dtype=torch.long),
                     cls=torch.as_tensor(target['cls'][:, None], device='cuda', dtype=torch.float32),
                     bboxes=torch.as_tensor(target['bboxes'], device='cuda'))
        records = {}
        for mode in ('history', 'last', 'mean'):
            bb = copy.deepcopy(ordinary)
            arm = copy.deepcopy(branch)
            # Nonzero diagnostic readout exposes all upstream derivatives; no
            # learning and no promotion of these diagnostic parameters.
            with torch.no_grad(): arm.output.weight.fill_(1e-4)
            arm.zero_grad(set_to_none=True); bb.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); tick = time.monotonic()
            modified, correction = arm(sequence, bb, observed_frames, fps, mode)
            pred = detection_training_predictions(head, [modified, *current[1:]])
            loss_fn = v8DetectionLoss(detector, tal_topk=1)
            centers, strides = make_anchors(pred['feats'], head.stride, .5)
            loss_fn.bce = CrowdAwareBCE(centers * strides,
                            torch.as_tensor(target['crowd_xyxy'], device='cuda'),
                            torch.as_tensor(target['crowd_cls'], device='cuda'), head.nc)
            loss, parts = loss_fn(pred, batch); loss = loss.sum()
            assert torch.isfinite(loss)
            loss.backward(); torch.cuda.synchronize()
            grads = [x.grad for x in list(arm.parameters()) + list(bb.parameters())]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert arm.input.weight.grad.norm() > 0
            records[mode] = dict(loss=float(loss.detach()), seconds=time.monotonic()-tick,
                            adapter_parameters=sum(x.numel() for x in arm.parameters()),
                            backend_parameters=sum(x.numel() for x in bb.parameters()),
                            correction_max=float(correction.detach().abs().max()),
                            adapter_gradient_norm=float(arm.input.weight.grad.norm()),
                            optimizer_steps=0)
            del bb, arm, pred, loss, loss_fn
        unchanged = all(torch.equal(before[n], x) for n, x in detector.named_parameters())
        assert unchanged
        service.write(FOLDER / 'report.json', dict(passed=True, complete=True,
                      full_video=service.read(FOLDER / 'video_completed.json'), controls=records,
                      detector_unchanged=unchanged, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      target_count=len(target['cls']), crowd_count=len(target['crowd_cls']),
                      training_started=False, optimizer_steps=0, scientific_verdict=None,
                      detector_accuracy_measured=False, hota_measured=False,
                      limitations=p['limitations'], updated=service.now()))
    faulthandler.cancel_dump_traceback_later()
    script = 'scripts/record_b2_causal960_20261028.py'
    target = RUN / 'b2_causal960_record'
    service.write(RUN / 'ready_jobs/114_b2_causal960_record.json', dict(name='b2_causal960_record',
                  folder=target.relative_to(ROOT).as_posix(), script=script,
                  script_sha256=service.sha(ROOT / script), protocol_sha256=service.sha(target / 'protocol.json')))


if __name__ == '__main__': run()
