"""Identity and complete writes for detector caches; legacy reading is separate."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weights_identity(model_path):
    path = Path(model_path).resolve(strict=True)
    return dict(model=str(path), weights_sha256=file_sha256(path))


def frame_inventory(frames):
    return dict(frame_names=np.asarray([p.name for p in frames]),
                source_frame_sha256=np.asarray([file_sha256(p) for p in frames]))


def verify_frame_inventory(cache, inventory):
    for key, value in inventory.items():
        if key not in cache or not np.array_equal(cache[key], value):
            raise RuntimeError('Source frame inventory/content differs from saved cache; use a new cache directory')


def verify_weights(model_path, identity):
    if weights_identity(model_path) != identity:
        raise RuntimeError('Weights changed while building cache; use a stable checkpoint and a new cache directory')


def ensure_cache_identity(cache_dir, metadata):
    """Do not infer provenance for pre-existing arrays from today's weights."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / 'meta.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if old.get('cache_schema') != metadata['cache_schema'] or 'weights_sha256' not in old:
            raise RuntimeError('Legacy cache has no verified weights identity. Read it explicitly as a saved result, or build in a new cache directory')
        for key, value in metadata.items():
            if key != 'date' and old.get(key) != value:
                raise RuntimeError(f'Cache identity/protocol differs for {key}; use a new cache directory')
    else:
        if any(cache_dir.rglob('*.npz')):
            raise RuntimeError('Existing cache has no metadata; refusing to assign provenance. Use a new cache directory')
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive create: a competing builder must not silently replace meta.
        with meta_path.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)


def validate_detection_arrays(cache, *, required=()):
    for key in ('xyxy', 'conf', 'cls', 'frame_idx', 'frame_names', *required):
        if key not in cache:
            raise ValueError(f'Incomplete detection cache: missing {key}')
    names = np.asarray(cache['frame_names'])
    boxes = np.asarray(cache['xyxy'])
    if names.ndim != 1 or names.dtype.kind not in 'US' or len(set(names.tolist())) != len(names):
        raise ValueError('Invalid or duplicate frame names')
    n = len(boxes)
    if boxes.shape != (n, 4) or not np.isfinite(boxes).all():
        raise ValueError('Invalid detection boxes')
    for key in ('conf', 'cls', 'frame_idx', *required, *(k for k in cache if k.startswith('q_') or k == 'stride')):
        values = np.asarray(cache[key])
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f'Invalid detection cache array: {key}')
    indices = np.asarray(cache['frame_idx'])
    if indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(names)):
        raise ValueError('Detection frame index outside saved frame inventory')
    confidence = np.asarray(cache['conf'])
    classes = np.asarray(cache['cls'])
    if np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError('Detection confidence must lie in [0,1]')
    if np.any(classes < 0) or np.any(classes != np.floor(classes)):
        raise ValueError('Detection class IDs must be nonnegative integers')
    if np.any(boxes[:,2:] < boxes[:,:2]):
        raise ValueError('Detection boxes have negative size')
    return cache


def load_detection_arrays(path, *, required=()):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return validate_detection_arrays(arrays, required=required)


def atomic_save_detection_arrays(path, arrays):
    """A failed save never leaves a final-looking partial .npz file."""
    validate_detection_arrays(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
