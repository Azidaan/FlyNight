"""Один причинный интерфейс сохранённых и живых наблюдений, без разметки."""
import numpy as np

VARIANTS = ('0', '1', '3', '4', '13', '14', '34', '134', 'mean', '1mean')


def _last(values, valid):
    order = np.where(valid, np.arange(valid.shape[1])[None], -1).max(1)
    result = values[np.arange(len(values)), np.maximum(order, 0)].copy()
    result[order < 0] = 0
    return result, order >= 0


def _collapse(values, valid, mode):
    if mode == 'last':
        summary, available = _last(values, valid)
    else:
        available = valid.any(1)
        summary = (np.where(valid[..., None], values, 0).sum(1) /
                   np.maximum(valid.sum(1, keepdims=True), 1))
    return np.repeat(summary[:, None], valid.shape[1], axis=1).astype(np.float32), np.repeat(available[:, None], valid.shape[1], axis=1)


def make_features(*, frame, box, geometry, image, current_embedding,
                  history_frame, history_box, history_quality, history_valid,
                  history_embedding, current_camera, history_camera,
                  current_epoch, history_epoch, video_size, variant):
    """Время — исходные кадры; перенос камеры — координаты320×180.

    frame/ID/det используются снаружи только как указатели. Абсолютные
    координаты, ID и метки личности не входят в результат.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    valid = np.asarray(history_valid, bool)
    frame = np.asarray(frame)
    if np.any(valid & ((history_frame >= frame[:, None]) | (frame[:, None]-history_frame > 60))):
        raise ValueError('История должна быть строго прошлой и не старше60кадров')
    if np.any(np.diff(np.where(valid, history_frame, np.iinfo(np.int32).max), axis=1) < 0):
        raise ValueError('История должна быть упорядочена и дополнена справа')
    boxes = np.asarray(box, np.float32)
    extent = np.maximum(boxes[:, 2:]-boxes[:, :2], 1)
    center = (boxes[:, :2]+boxes[:, 2:])/2
    delta_camera = np.asarray(current_camera)[:, None]-np.asarray(history_camera)
    size = np.asarray(video_size, np.float32)
    if size.ndim == 1:
        size = np.broadcast_to(size, (len(boxes), 2))
    shift = delta_camera*size[:, None]/np.array([320., 180.], np.float32)
    past = np.asarray(history_box, np.float32).copy()
    past[..., :2] += shift
    past[..., 2:] += shift
    reliable = valid & (history_epoch == current_epoch[:, None])
    past_extent = np.maximum(past[..., 2:]-past[..., :2], 1)
    past_center = (past[..., :2]+past[..., 2:])/2
    ages = np.clip((frame[:, None]-history_frame)/60., 0, 1).astype(np.float32)
    quality = np.asarray(history_quality, np.float32)
    observations = np.concatenate([
        (past_center-center[:, None])/extent[:, None],
        np.log(past_extent/extent[:, None]), ages[..., None], quality[..., None],
        reliable[..., None], valid[..., None]], axis=-1).astype(np.float32)
    observations = np.clip(np.where(reliable[..., None], observations, 0), -10, 10)
    embedding = np.asarray(history_embedding, np.float32)
    history_appearance = np.concatenate([embedding, ages[..., None], quality[..., None]], axis=-1)
    mode = 'mean' if 'mean' in variant else 'last'
    if '3' not in variant:
        history_appearance, app_valid = _collapse(history_appearance, valid, mode)
    else:
        app_valid = valid.copy()
    emb = history_appearance[..., :256]
    emb = emb/np.maximum(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-8)
    current = np.asarray(current_embedding, np.float32)[:, None]
    appearance = np.concatenate([current-emb, (current*emb).sum(-1, keepdims=True),
        history_appearance[..., 256:], app_valid[..., None]], axis=-1).astype(np.float32)
    appearance = np.where(app_valid[..., None], appearance, 0)
    if '4' not in variant:
        observations, obs_valid = _collapse(observations, reliable, mode)
    else:
        obs_valid = reliable
    pixels = np.asarray(image, np.float32)/255.
    pixels = pixels.copy()
    if '1' not in variant:
        pixels[:, 1] = 0  # Тот же текущий серый кадр и маска, без давности движения.
    result = dict(image=pixels, geometry=np.asarray(geometry, np.float32).copy(),
                  appearance=appearance, observations=observations,
                  appearance_valid=app_valid, observation_valid=obs_valid)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Нечисловой вход')
    return result


def from_cache(inputs, index, embeddings, images, rows, variant):
    """Только inputs/index; targets и истинные личности сюда не передаются."""
    rows = np.asarray(rows, np.int64)
    current = index['current_index'][rows]
    hi = index['history_index'][rows]
    safe = np.maximum(hi, 0)
    return make_features(frame=inputs['frame'][rows], box=inputs['box'][rows],
        geometry=inputs['geometry'][rows], image=images[current],
        current_embedding=embeddings[current], history_frame=inputs['history_frame'][rows],
        history_box=inputs['history_box'][rows], history_quality=inputs['history_quality'][rows],
        history_valid=inputs['history_valid'][rows], history_embedding=embeddings[safe],
        current_camera=index['camera'][current], history_camera=index['camera'][safe],
        current_epoch=index['camera_epoch'][current], history_epoch=index['camera_epoch'][safe],
        video_size=index['video_size'], variant=variant)
