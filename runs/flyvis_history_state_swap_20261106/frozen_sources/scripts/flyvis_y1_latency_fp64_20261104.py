"""Я1: замороженная штатная FlyVis, известные переходы движения, без обучения."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/flyvis_y1_latency_fp64_20261104'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.next')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def status(**data):
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    data['disk_free_bytes'] = __import__('shutil').disk_usage(OUT).free
    assert data['disk_free_bytes'] >= 5_000_000_000, '5 GB reserve'
    write('status.json', data)


def protocol():
    p = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in p['frozen_sources'].items():
        assert sha(ROOT / name) == expected, name
    assert datetime.now(timezone.utc).timestamp() < datetime.fromisoformat(p['hard_stop']).timestamp()
    return p


def positions(case):
    import numpy as np
    v = np.zeros((79, 2), np.int32)
    v[31:55, case['axis']] = case['speed']
    if case['event'] == 'reverse':
        v[55:, case['axis']] = -case['speed']
    return np.cumsum(v, axis=0), v


def frames_for(gray, case):
    import numpy as np
    pos, _ = positions(case)
    h, w = gray.shape
    y0, x0 = (h - 512) // 2, (w - 512) // 2
    result = []
    for dx, dy in pos:
        y, x = y0 - int(dy), x0 - int(dx)
        assert min(x, y) >= 0 and x + 512 <= w and y + 512 <= h
        crop = gray[y:y + 512, x:x + 512]
        result.append(np.rint(np.clip(127.5 + case['contrast'] * (crop.astype(np.float32) - 127.5), 0, 255)).astype(np.uint8))
    return np.stack(result)


def cpu_eye(frames, centers):
    import cv2
    import numpy as np
    xy = np.asarray(centers, np.int64) + 195
    return np.stack([cv2.boxFilter(cv2.resize(g, (391, 391), interpolation=cv2.INTER_AREA).astype(np.float32) / 255,
                                   -1, (13, 13), normalize=True, borderType=cv2.BORDER_CONSTANT)[xy[:, 0], xy[:, 1]] for g in frames])


def prepare():
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    p = protocol()
    texture_path = OUT / 'textures.npz'
    if not texture_path.exists():
        data, sources = {}, []
        for i, video in enumerate(p['videos']):
            status(stage='prepare_textures', done=i, total=4)
            path = ROOT / 'data/bdd100k/videos/train' / (video + '.mov')
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            assert ok, path
            data[str(i)] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sources.append(dict(video=video, frame=0, source_fps=fps, video_sha256=sha(path),
                                gray_sha256=hashlib.sha256(data[str(i)].tobytes()).hexdigest()))
        np.savez_compressed(texture_path, **data)
        write('textures.json', dict(sources=sources, sha256=sha(texture_path)))
    gray = np.load(texture_path)
    # Точные целочисленные координаты штатного BoxEye, независимое усреднение OpenCV.
    centers = np.array([(int(13 * (u + v / 2)), 13 * v) for u in range(-15, 16)
                        for v in range(max(-15, -15 - u), min(15, 15 - u) + 1)], np.int64)
    mask = (centers.astype(float) ** 2).sum(1) <= (195 * .5) ** 2
    # Координаты выборки DIS после того же пространственного преобразования.
    pixels = np.rint((centers + 195 + .5) * 512 / 391 - .5).astype(int)
    np.savez_compressed(OUT / 'geometry.npz', centers=centers, mask=mask, pixels=pixels)
    for i, case in enumerate(p['cases']):
        target = OUT / 'inputs' / (case['id'] + '.npz')
        marker = target.with_suffix('.json')
        if marker.exists():
            record = json.loads(marker.read_text())
            assert sha(target) == record['sha256']
            continue
        status(stage='prepare_inputs', done=i, total=len(p['cases']), case=case['id'])
        frames = frames_for(gray[str(case['scene'])], case)
        eye = cpu_eye(frames, centers)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis_flow = np.zeros((len(frames), mask.sum(), 2), np.float32)
        for j in range(1, len(frames)):
            f = dis.calc(frames[j - 1], frames[j], None)
            dis_flow[j] = f[pixels[mask, 0], pixels[mask, 1]]
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, frames=frames, eye_reference=eye, dis=dis_flow)
        write('inputs/' + case['id'] + '.json', dict(sha256=sha(target),
              frame_sha256=[hashlib.sha256(g.tobytes()).hexdigest() for g in frames]))
        if i == 0:
            required = 2 * target.stat().st_size * (len(p['cases']) - 1) + 5_000_000_000
            assert __import__('shutil').disk_usage(OUT).free > required
            write('input_disk_gate.json', dict(required_bytes=required, passed=True))
    write('inputs_completed.json', dict(complete=True, cases=len(p['cases']), protocol_sha256=sha(OUT / 'protocol.json')))
    status(stage='inputs_completed', done=len(p['cases']), total=len(p['cases']))


def parameter_sha(net, decoder):
    h = hashlib.sha256()
    for module in (net, decoder):
        for name, tensor in module.state_dict().items():
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer():
    requested_device = os.environ.get('Y1_EXECUTION_DEVICE', 'cpu')
    if requested_device == 'cpu':os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import cv2
    import torch
    import importlib
    from contextlib import nullcontext
    from tracking.flyvis_adapter import activate_flyvis
    fv=activate_flyvis()
    if requested_device == 'cpu':
        for m in [fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')]:m.device=torch.device('cpu')
    torch.set_default_device(requested_device)
    torch.set_default_dtype(torch.float64)
    from tracking.flyvis_motion import FlyVisMotionEstimator
    from tracking.gpu_lock import GpuLock
    p = protocol()
    assert json.loads((OUT / 'inputs_completed.json').read_text())['complete']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with (GpuLock('Local\\DissertationSingleGpu') if requested_device == 'cuda' else nullcontext()):
        model = FlyVisMotionEstimator(mode='full', source_fps=30, flow_gain=1, temporal_resampling='hold', subtract_initial_baseline=False)
        model.network.double();model.decoder.double();model.eye.conv.double()
        net, dec, eye, dev = model.network, model.decoder, model.eye, model.flyvis.device
        assert str(dev) == requested_device and model.integration_steps == 2
        assert model.eye_height == model.eye_width == 391
        initial_hash = parameter_sha(net, dec)
        # clamp — часть штатного forward. Его первое действие фиксируется отдельно.
        net.clamp()
        frozen_hash = parameter_sha(net, dec)
        geo = np.load(OUT / 'geometry.npz')
        assert np.array_equal(eye.receptor_centers.cpu().numpy(), geo['centers'])
        mask = geo['mask']
        stage_idx = {name: np.asarray(net.stimulus.layer_index[name]) for name in ['R1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']}
        write('model_loaded.json', dict(device=str(dev), initial_parameter_sha=initial_hash, frozen_parameter_sha=frozen_hash,
              default_clamp_changed_parameters=initial_hash != frozen_hash, dt=model.dt, steps=model.integration_steps,
              modes=p['modes'], stages=list(stage_idx), tf32=False))
        if requested_device == 'cuda':torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            for ci, case in enumerate(p['cases']):
                target = OUT / 'outputs' / (case['id'] + '.npz')
                marker = target.with_suffix('.json')
                if marker.exists():
                    record = json.loads(marker.read_text())
                    assert sha(target) == record['sha256'] and record['protocol_sha256'] == sha(OUT / 'protocol.json')
                    continue
                free = torch.cuda.mem_get_info()[0] if requested_device == 'cuda' else __import__('psutil').virtual_memory().available
                assert free >= 1_500_000_000, 'Insufficient free GPU memory'
                inp = np.load(OUT / 'inputs' / (case['id'] + '.npz'))
                rendered = []
                for chunk in range(0, 79, 8):
                    cart = np.stack([cv2.resize(g, (391, 391), interpolation=cv2.INTER_AREA) for g in inp['frames'][chunk:chunk + 8]])
                    x = torch.from_numpy(cart.astype(np.float64) / 255).to(dev)[None]
                    rendered.append(eye(x)[0, :, 0].cpu().numpy())
                hexes = np.concatenate(rendered)
                eye_diff = float(np.max(np.abs(hexes - inp['eye_reference'])))
                assert eye_diff <= 1e-5, eye_diff
                # Два независимых состояния для hold/linear; один и тот же набор признаков глаза.
                flow = np.empty((2, 2, 79, 721, 2), np.float64)
                activity = np.empty((2, 79, len(stage_idx)), np.float64)
                first = torch.from_numpy(hexes[0]).to(dev)[None, None]
                state = net.fade_in_state(1.0, model.dt, first.repeat(2, 1, 1))
                init = dec(state.nodes.activity[:, None])[:, 0].cpu().numpy().transpose(0, 2, 1)
                flow[:, :, 0] = init[:, None] * np.array([512 / 169, -512 / 169], np.float64)
                activity[:, 0] = np.stack([state.nodes.activity[:, idx[mask]].mean(1).cpu().numpy() for idx in stage_idx.values()], axis=1)
                last = torch.from_numpy(hexes[0]).to(dev)[None, None, None]
                for j in range(1, 79):
                    current = torch.from_numpy(hexes[j]).to(dev)[None, None, None]
                    hold = current.expand(1, 2, 1, 721)
                    alpha = torch.tensor([.5, 1.], device=dev).view(1, 2, 1, 1)
                    linear = last * (1 - alpha) + current * alpha
                    stim = torch.cat([hold, linear], dim=0)
                    states = net.simulate(stim, model.dt, initial_state=state, as_states=True)
                    act = torch.stack([s.nodes.activity for s in states], dim=1)
                    decoded = dec(act)
                    field = torch.stack([decoded.mean(1), decoded[:, -1]], dim=1)
                    flow[:, :, j] = field.cpu().numpy().transpose(0, 1, 3, 2) * np.array([512 / 169, -512 / 169], np.float64)
                    state = states[-1]
                    activity[:, j] = np.stack([state.nodes.activity[:, idx[mask]].mean(1).cpu().numpy() for idx in stage_idx.values()], axis=1)
                    last = current
                    if j % 10 == 0:
                        status(stage='inference', done=ci, total=len(p['cases']), case=case['id'], frame=j, frames=79,
                               elapsed_seconds=time.monotonic() - started)
                assert np.isfinite(flow).all() and np.isfinite(activity).all()
                target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target, flow=flow, activity=activity, hexes=hexes)
                if requested_device == 'cuda':torch.cuda.synchronize()
                write('outputs/' + case['id'] + '.json', dict(sha256=sha(target), protocol_sha256=sha(OUT / 'protocol.json'),
                      input_sha256=sha(OUT / 'inputs' / (case['id'] + '.npz')), eye_max_diff=eye_diff,
                      parameter_sha256=parameter_sha(net, dec)))
                assert parameter_sha(net, dec) == frozen_hash
                if ci == 0:
                    seconds = time.monotonic() - started
                    saved = sum(q.stat().st_size for q in OUT.rglob('*.npz'))
                    required = 2 * target.stat().st_size * (len(p['cases']) - 1) + 5_000_000_000
                    assert __import__('shutil').disk_usage(OUT).free >= required
                    write('first_complete_case_price.json', dict(seconds=seconds, remaining_cases=len(p['cases']) - 1,
                          estimate_remaining_seconds=seconds * (len(p['cases']) - 1), time_is_forecast_not_cap=True,
                          saved_bytes_so_far=saved, required_remaining_bytes=required, disk_gate_passed=True))
            write('inference_completed.json', dict(complete=True, cases=len(p['cases']), modes=2, training=False,
                  frozen_parameters_unchanged=parameter_sha(net, dec) == frozen_hash,
                  elapsed_seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated() if requested_device == 'cuda' else 0))
        status(stage='inference_completed', done=len(p['cases']), total=len(p['cases']))


def reference():
    import numpy as np
    import torch
    from tracking.flyvis_motion import FlyVisMotionEstimator
    p = protocol()
    torch.set_num_threads(2)
    torch.set_default_device('cpu')
    model = FlyVisMotionEstimator(mode='full', source_fps=30, flow_gain=1,
                                 temporal_resampling='hold', subtract_initial_baseline=False)
    assert str(model.flyvis.device) == 'cpu' and not torch.cuda.is_initialized()
    case = p['cases'][0]
    inp = np.load(OUT / 'inputs' / (case['id'] + '.npz'))
    gpu = np.load(OUT / 'outputs' / (case['id'] + '.npz'))
    values, diffs = [], []
    for j, image in enumerate(inp['frames']):
        model.observe(image)
        if j:
            d = model.last_decoded.cpu().numpy()[0]
            both = np.stack([d.mean(0), d[-1]], axis=0).transpose(0, 2, 1) * np.array([512 / 169, -512 / 169], np.float32)
            diffs.append(float(np.max(np.abs(both - gpu['flow'][0, :, j]))))
            values.append(both)
        if j % 10 == 0:status(stage='cpu_reference', frame=j, total=79)
    np.savez_compressed(OUT / 'cpu_reference.npz', flow=np.stack(values))
    write('cpu_reference.json', dict(complete=True, passed=max(diffs) <= 1e-5, max_abs_diff=max(diffs),
          tolerance=1e-5, case=case['id'], frames_checked=78, mode='hold',
          scope='One complete fixed case via streaming estimator on CPU; not full CPU replay of all network responses',
          cuda_initialized=torch.cuda.is_initialized()))
    assert max(diffs) <= 1e-5, max(diffs)


if __name__ == '__main__':
    a = argparse.ArgumentParser();a.add_argument('--stage', choices=['prepare', 'infer', 'reference'], required=True)
    args = a.parse_args()
    if args.stage != 'infer':os.environ['CUDA_VISIBLE_DEVICES'] = ''
    {'prepare': prepare, 'infer': infer, 'reference': reference}[args.stage]()
