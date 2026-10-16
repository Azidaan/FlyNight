"""Shared immutable frame features; original per-arm RoI operations unchanged."""
from pathlib import Path
from datetime import datetime
import json
import time
import zipfile
import numpy as np


def tasks(path):
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in ('track_id', 'start', 'end', 'first_box', 'last_box')}
    size = len(arrays['track_id'])
    need = {}
    for k in range(size):
        need.setdefault(int(arrays['start'][k]), []).append((k, 0, arrays['first_box'][k]))
        need.setdefault(int(arrays['end'][k]), []).append((k, 1, arrays['last_box'][k]))
    return size, need


def extract_video(model, video_path, signatures, outputs, progress=None, check_first_frame=True):
    import cv2
    import torch
    from tracking.masa_r50 import preprocess
    from tracking.z13_checkpoint import replace_with_retry
    assert not model.training and not torch.is_autocast_enabled()
    jobs = {arm: tasks(path) for arm, path in signatures.items()}
    embeddings = {arm: np.zeros((size, 2, 256), np.float16) for arm, (size, _) in jobs.items()}
    written = {arm: np.zeros((size, 2), bool) for arm, (size, _) in jobs.items()}
    union = set().union(*(set(need) for _, need in jobs.values()))
    last = max(union) if union else -1
    stats = {'feature_calls': 0, 'feature_calls_without_sharing': sum(len(n) for _, n in jobs.values()),
             'guard_feature_calls': 0, 'frames_decoded': 0, 'arms': list(jobs)}
    cap = cv2.VideoCapture(str(video_path))
    began = time.perf_counter(); frame_index = 0
    guard_done = False
    with torch.no_grad():
        try:
            while frame_index <= last:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Video ended before requested frame {last}')
                if frame_index in union:
                    image, scale = preprocess(frame)
                    features = model.features(image)
                    stats['feature_calls'] += 1
                    reference = None
                    if check_first_frame and not guard_done:
                        reference = model.features(image)
                        stats['guard_feature_calls'] += 1
                    for arm, (_, need) in jobs.items():
                        if frame_index not in need:
                            continue
                        items = need[frame_index]
                        boxes = torch.tensor(np.stack([b for _, _, b in items]).astype(np.float32), device='cuda')
                        e = torch.nn.functional.normalize(model.embed_boxes(features, boxes, scale), dim=1).cpu().numpy()
                        if reference is not None:
                            r = torch.nn.functional.normalize(model.embed_boxes(reference, boxes, scale), dim=1).cpu().numpy()
                            if not np.array_equal(e.astype(np.float16), r.astype(np.float16)):
                                raise RuntimeError('Same-frame fp16 embedding parity guard failed')
                        for (k, side, _), value in zip(items, e):
                            embeddings[arm][k, side] = value
                            written[arm][k, side] = True
                    guard_done = True
                frame_index += 1
                if frame_index % 50 == 0:
                    print(f'[SHARED MASA] {video_path.stem}: {frame_index} frames, {stats["feature_calls"]} frame feature calls', flush=True)
                    if progress:
                        progress(frame_index, last + 1)
        finally:
            cap.release()
    assert all(mask.all() for mask in written.values()), 'Unwritten embedding slots'
    assert all(np.isfinite(e).all() for e in embeddings.values())
    for arm, output in outputs.items():
        output = Path(output)
        assert not output.exists(), 'Refusing to overwrite existing embeddings'
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.next.npz')
        assert not temporary.exists(), 'Preserve interrupted temporary file before retry'
        np.savez_compressed(temporary, first=embeddings[arm][:, 0], last=embeddings[arm][:, 1])
        replace_with_retry(temporary, output)
    stats.update(seconds=time.perf_counter() - began, frames_decoded=frame_index,
                 parity_guard_checked=guard_done if check_first_frame else False)
    return stats


def run_full(run):
    import torch
    from tracking.masa_r50 import build
    from tracking.z13_checkpoint import atomic_json
    torch.set_num_threads(2)
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    videos = protocol['videos']; arms = protocol['arms']
    model = build()
    for index, video in enumerate(videos):
        signatures = {arm: run / arm / 'signatures' / video / 'signatures.npz' for arm in arms}
        outputs = {arm: run / arm / 'masa' / f'{video}.npz' for arm in arms}
        # Validate even a partially published set before deciding which arms to skip.
        for arm, output in outputs.items():
            temporary = output.with_name(output.name + '.next.npz')
            for candidate in (output, temporary):
                if not candidate.exists():
                    continue
                valid = False
                if candidate == output:
                    try:
                        size, _ = tasks(signatures[arm])
                        with np.load(candidate) as e:
                            valid = e['first'].shape == e['last'].shape == (size, 256) and np.isfinite(e['first']).all() and np.isfinite(e['last']).all()
                    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                        valid = False
                if not valid:
                    candidate.resolve().relative_to(run.resolve())
                    archive = candidate.with_name(candidate.name + '.partial_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    assert not archive.exists()
                    candidate.rename(archive)
        if all(path.exists() for path in outputs.values()):
            for arm in arms:
                size, _ = tasks(signatures[arm])
                with np.load(outputs[arm]) as e:
                    assert e['first'].shape == e['last'].shape == (size, 256)
                    assert np.isfinite(e['first']).all() and np.isfinite(e['last']).all()
        else:
            pending = {arm: path for arm, path in signatures.items() if not outputs[arm].exists()}
            def progress(frame, total):
                atomic_json(run / 'shared_masa_status.json', {'state': 'running', 'video': video,
                    'completed_videos': index, 'total_videos': len(videos), 'frame': frame, 'frames': total})
            # If a prior interrupted save completed one arm, keep it and compute only missing arms.
            info = extract_video(model, Path(__file__).resolve().parents[1] / 'data/bdd100k/videos/train' / f'{video}.mov',
                                 pending, {arm: outputs[arm] for arm in pending}, progress)
            atomic_json(run / 'shared_masa_details' / f'{video}.json', info)
        for arm in arms:
            size, _ = tasks(signatures[arm])
            compatibility = run / arm / 'dino_compat' / f'{video}.npz'
            valid = False
            if compatibility.exists():
                try:
                    with np.load(compatibility) as e:
                        valid = all(e[k].shape == (size, 384) and not np.any(e[k]) for k in ('first', 'last'))
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                    pass
                if not valid:
                    compatibility.resolve().relative_to(run.resolve())
                    compatibility.rename(compatibility.with_name(compatibility.name + '.partial_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            if not compatibility.exists():
                from tracking.z13_checkpoint import replace_with_retry
                compatibility.parent.mkdir(parents=True, exist_ok=True)
                temporary = compatibility.with_name(compatibility.name + '.next.npz')
                if temporary.exists():
                    temporary.rename(temporary.with_name(temporary.name + '.partial_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
                np.savez_compressed(temporary, first=np.zeros((size, 384), np.float16), last=np.zeros((size, 384), np.float16))
                replace_with_retry(temporary, compatibility)
        atomic_json(run / 'shared_masa_status.json', {'state': 'running', 'completed_videos': index + 1,
                    'total_videos': len(videos), 'video': video})
        print(f'[SHARED SAVED] {index + 1}/{len(videos)} {video}, all arms', flush=True)
    atomic_json(run / 'shared_masa_status.json', {'state': 'completed', 'completed_videos': len(videos)})
