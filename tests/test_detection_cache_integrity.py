"""Cache identity and frame-binding failures must not become valid data."""
from pathlib import Path
from types import SimpleNamespace
import json
import glob

import numpy as np
import pytest
import torch

from tracking import detection_cache as cache


@pytest.fixture
def setup_cache(tmp_path, monkeypatch):
    weights = tmp_path / 'model.pt'
    weights.write_bytes(b'original-weights')
    image_root = tmp_path / 'data/images/track/val/v'
    image_root.mkdir(parents=True)
    for name in ('000.jpg', '001.jpg'):
        (image_root / name).write_bytes(b'fake-image')
    monkeypatch.setattr(cache, 'DATA_ROOT', tmp_path / 'data')
    state = dict(calls=0, mode='normal')

    class Model:
        def __init__(self, path):
            state['calls'] += 1

        def predict(self, source, **kwargs):
            if isinstance(source, list):
                paths = [Path(p) for p in source]
            else:
                # Mirrors the installed loader's directory behavior: every
                # supported image format, unlike the builder's JPG inventory.
                paths = [Path(p) for p in sorted(glob.glob(source if '*' in source else str(Path(source)/'*')))]
            if state['mode'] == 'short':
                paths = paths[:1]
            if state['mode'] == 'reversed':
                paths = paths[::-1]
            for i, p in enumerate(paths):
                boxes = SimpleNamespace(xyxy=torch.tensor([[i, 1, i+2, 3.]]),
                                        conf=torch.tensor([.8]), cls=torch.tensor([2.]))
                class Boxes:
                    xyxy, conf, cls = boxes.xyxy, boxes.conf, boxes.cls
                    def __len__(self): return 1
                yield SimpleNamespace(path=str(p), boxes=Boxes())

    monkeypatch.setattr('ultralytics.YOLO', Model)
    folder = tmp_path / 'cache'
    def build():
        return cache.build_detection_cache(str(weights), ['v'], cache_dir=folder, device='cpu')
    return weights, image_root, folder, state, build


def test_changed_weights_same_path_rejected_without_touching_cache(setup_cache):
    weights, _, folder, state, build = setup_cache
    build()
    before = (folder/'val/v.npz').read_bytes()
    weights.write_bytes(b'replaced-weights')
    with pytest.raises(RuntimeError, match='(?i)(weight|вес|identity|протокол)'):
        build()
    assert state['calls'] == 1
    assert (folder/'val/v.npz').read_bytes() == before


def test_changed_threshold_rejected(setup_cache, monkeypatch):
    _, _, _, state, build = setup_cache
    build()
    monkeypatch.setattr(cache, 'CACHE_CONF', .2)
    with pytest.raises(RuntimeError):
        build()
    assert state['calls'] == 1


@pytest.mark.parametrize('mode', ['short', 'reversed'])
def test_wrong_result_binding_never_written_as_complete(setup_cache, mode):
    _, _, folder, state, build = setup_cache
    state['mode'] = mode
    with pytest.raises(RuntimeError, match='(?i)(frame|кадр)'):
        build()
    assert not (folder/'val/v.npz').exists()


def test_inventory_and_detector_source_cannot_disagree(setup_cache):
    _, images, folder, _, build = setup_cache
    (images/'000a.png').write_bytes(b'unsupported-by-inventory')
    build()
    data = cache.load_video_cache(folder, 'val', 'v')
    assert data['frame_names'].tolist() == ['000.jpg', '001.jpg']
    assert data['frame_idx'].tolist() == [0, 1]


def test_missing_metadata_never_adopts_existing_cache(setup_cache):
    _, _, folder, _, build = setup_cache
    build()
    (folder/'meta.json').unlink()
    with pytest.raises(RuntimeError):
        build()
    assert not (folder/'meta.json').exists()


def test_legacy_cache_remains_readable_but_is_not_certified(setup_cache):
    weights, _, folder, _, build = setup_cache
    build()
    (folder/'meta.json').write_text(json.dumps(dict(model=str(weights), imgsz=640)), encoding='utf-8')
    assert len(list(cache.iter_frames(cache.load_video_cache(folder, 'val', 'v')))) == 2
    with pytest.raises(RuntimeError):
        build()


def test_valid_reuse_does_not_load_model_again(setup_cache):
    _, _, folder, state, build = setup_cache
    build()
    before = (folder/'val/v.npz').read_bytes()
    build()
    assert state['calls'] == 1
    assert (folder/'val/v.npz').read_bytes() == before


def test_corrupt_existing_cache_is_not_silently_skipped(setup_cache):
    _, _, folder, _, build = setup_cache
    build()
    (folder/'val/v.npz').write_bytes(b'interrupted-npz')
    with pytest.raises((RuntimeError, ValueError, OSError)):
        build()


def test_failed_atomic_save_preserves_original(tmp_path, monkeypatch):
    from tracking import cache_integrity as integrity
    path = tmp_path/'prior.npz'
    arrays = dict(xyxy=np.zeros((0,4)), conf=np.zeros(0), cls=np.zeros(0),
                  frame_idx=np.zeros(0, np.int32), frame_names=np.array(['a.jpg']))
    integrity.atomic_save_detection_arrays(path, arrays)
    before = path.read_bytes()
    def interrupted(stream, **arrays):
        stream.write(b'partial')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(integrity.np, 'savez_compressed', interrupted)
    with pytest.raises(OSError, match='disk failure'):
        integrity.atomic_save_detection_arrays(path, arrays)
    assert path.read_bytes() == before
    assert list(tmp_path.glob('*.part')) == []


def test_q_cache_protocol_and_empty_detections(setup_cache, monkeypatch):
    from tracking import inference_with_q as q
    weights, images, folder, _, _ = setup_cache
    monkeypatch.setattr(q, 'DATA_ROOT', images.parents[3])
    class Q:
        with_drb = True
        def __init__(self, *args, **kwargs): pass
        def predict_frame(self, image, aggregations):
            return dict(xyxy=np.zeros((0,4),np.float32),conf=np.zeros(0,np.float32),
                        cls=np.zeros(0,np.int16),stride=np.zeros(0,np.int16),
                        **{f'q_{a}':np.zeros(0,np.float32) for a in aggregations})
    monkeypatch.setattr(q, 'QInference', Q)
    def build(aggregations=('mean',)):
        return q.build_detection_cache_with_q(str(weights), ['v'], cache_dir=folder,
                                               aggregations=aggregations, device='cpu')
    build()
    saved = cache.load_video_cache(folder,'val','v')
    assert saved['q_mean'].shape == (0,)
    assert len(saved['frame_names']) == 2
    build()
    with pytest.raises(RuntimeError):
        build(('max',))
    monkeypatch.setattr(q,'IMGSZ',960)
    with pytest.raises(RuntimeError):
        build()


def test_real_q_inference_empty_frame_has_empty_quality_arrays():
    from tracking.inference_with_q import QInference
    qi = QInference.__new__(QInference)
    boxes = SimpleNamespace(xyxy=torch.zeros((0,4)),conf=torch.zeros(0),cls=torch.zeros(0))
    class EmptyBoxes:
        xyxy, conf, cls = boxes.xyxy, boxes.conf, boxes.cls
        def __len__(self): return 0
    result = SimpleNamespace(boxes=EmptyBoxes())
    qi.model = SimpleNamespace(predict=lambda **kwargs:[result])
    qi.conf=.01; qi.iou=.7; qi.device='cpu'; qi.with_drb=True
    qi.head=SimpleNamespace(_last_anchor_idx=torch.zeros((1,0,1),dtype=torch.int64),strides=torch.tensor([8]))
    out = qi.predict_frame('unused.jpg', ('mean','max'))
    assert out['q_mean'].shape == out['q_max'].shape == (0,)


@pytest.mark.parametrize('change', ['add', 'content', 'rename'])
def test_reuse_rejects_changed_source_frames(setup_cache, change):
    _, images, folder, _, build = setup_cache
    build()
    before = (folder/'val/v.npz').read_bytes()
    if change == 'add':
        (images/'002.jpg').write_bytes(b'new-frame')
    elif change == 'rename':
        (images/'001.jpg').rename(images/'003.jpg')
    else:
        (images/'001.jpg').write_bytes(b'changed-frame-content')
    with pytest.raises(RuntimeError, match='frame inventory'):
        build()
    assert (folder/'val/v.npz').read_bytes() == before


def test_brackets_in_source_parent_preserve_loader_semantics(setup_cache, monkeypatch):
    _, images, folder, _, build = setup_cache
    old = images.parents[3]
    new = old.with_name('data[1]')
    old.rename(new)
    monkeypatch.setattr(cache,'DATA_ROOT',new)
    build()
    assert len(cache.load_video_cache(folder,'val','v')['frame_names']) == 2


@pytest.mark.parametrize('key,value', [('conf',[-.1]),('conf',[1.1]),('cls',[1.5]),('cls',[-1]),('xyxy',[[3,0,2,1]])])
def test_invalid_numerical_detections_rejected(key,value):
    from tracking.cache_integrity import validate_detection_arrays
    arrays=dict(xyxy=np.array([[0,0,1,1]]),conf=np.array([.5]),cls=np.array([0]),
                frame_idx=np.array([0]),frame_names=np.array(['a.jpg']))
    arrays[key]=np.asarray(value)
    with pytest.raises(ValueError):
        validate_detection_arrays(arrays)
