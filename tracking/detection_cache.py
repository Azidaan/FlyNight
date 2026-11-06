"""Кэширование детекций для сеточного поиска параметров трекера (Этап 8).

Ключевая оптимизация плана (подэтап 8.2): детектор прогоняется по каждому
видео ровно один раз, рамки + confidence сохраняются в npz. Сеточный поиск
варьирует только этап ассоциации (tracking/offline_association.py), который
работает на CPU за секунды на конфигурацию — вместо полного прогона
детектора на каждую точку сетки.

Формат кэша: <cache_dir>/<split>/<video_id>.npz с плоскими массивами
    xyxy (N,4) float32, conf (N,) float32, cls (N,) int16,
    frame_idx (N,) int32 — номер кадра каждой детекции,
    frame_names (F,) unicode — имена jpg-кадров (для scalabel-вывода).
Мета: <cache_dir>/meta.json (модель, пороги NMS, дата).

Порог conf при кэшировании = 0.01: заведомо ниже минимального
track_low_thresh сетки, чтобы у ассоциации был полный материал.
На Этапе 9 в этот же кэш добавятся значения q_i (отдельным массивом).
"""
import json
import glob
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from tracking.cache_integrity import (atomic_save_detection_arrays, ensure_cache_identity,
    frame_inventory, load_detection_arrays, verify_frame_inventory, verify_weights, weights_identity)

CACHE_ROOT = PROJECT_ROOT / 'runs' / 'det_cache'
CACHE_CONF = 0.01   # NMS confidence при кэшировании (ниже любого порога сетки)
CACHE_IOU = 0.7     # NMS IoU — дефолт Ultralytics predict


def build_detection_cache(
    model_path: str,
    video_ids: list[str],
    split: str = 'val',
    cache_dir: str | Path = CACHE_ROOT,
    device: str | int = 0,
    imgsz: int = 640,
) -> Path:
    """Прогоняет детектор по видео и пишет npz-кэш. Пропускает готовые видео.

    Возвращает директорию кэша для split.
    """
    from ultralytics import YOLO

    cache_dir = Path(cache_dir)
    out_dir = cache_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = DATA_ROOT / 'images' / 'track' / split

    # imgsz пишется в мету намеренно: кэши на 640 и 960 внешне неразличимы
    # (рамки в обоих случаях в координатах исходного кадра), и перепутать их
    # значило бы сравнивать разные стенды, не заметив этого.
    identity = weights_identity(model_path)
    meta = {**identity, 'cache_schema': 'detection_jpg_v2', 'conf': CACHE_CONF, 'iou': CACHE_IOU,
            'imgsz': imgsz, 'data_root': str(DATA_ROOT.resolve()), 'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    ensure_cache_identity(cache_dir, meta)

    model = None
    for i, vid in enumerate(video_ids, 1):
        out_npz = out_dir / f'{vid}.npz'
        frames = sorted((img_root / vid).glob('*.jpg'))
        if not frames:
            raise ValueError(f'No JPG frames found for {vid}')
        inventory = frame_inventory(frames)
        if out_npz.exists():
            verify_frame_inventory(load_detection_arrays(out_npz), inventory)
            continue
        if model is None:
            model = YOLO(model_path)
        xyxy_all, conf_all, cls_all, fidx_all = [], [], [], []
        t0 = time.time()
        results = model.predict(source=glob.escape(str(img_root / vid)) + '/*.jpg', conf=CACHE_CONF,
                                iou=CACHE_IOU, device=device, stream=True,
                                imgsz=imgsz, verbose=False)
        seen = 0
        for fidx, res in enumerate(results):
            if fidx >= len(frames) or Path(res.path).resolve() != frames[fidx].resolve():
                raise RuntimeError(f'Detector result does not match expected frame {fidx} for {vid}')
            seen += 1
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            n = len(boxes)
            xyxy_all.append(boxes.xyxy.cpu().numpy().astype(np.float32))
            conf_all.append(boxes.conf.cpu().numpy().astype(np.float32))
            cls_all.append(boxes.cls.cpu().numpy().astype(np.int16))
            fidx_all.append(np.full(n, fidx, dtype=np.int32))
        if seen != len(frames):
            raise RuntimeError(f'Detector returned {seen} frames instead of {len(frames)} for {vid}')
        verify_weights(model_path, identity)
        verify_frame_inventory(frame_inventory(frames), inventory)
        atomic_save_detection_arrays(out_npz, dict(
            xyxy=np.concatenate(xyxy_all) if xyxy_all else np.zeros((0, 4), np.float32),
            conf=np.concatenate(conf_all) if conf_all else np.zeros(0, np.float32),
            cls=np.concatenate(cls_all) if cls_all else np.zeros(0, np.int16),
            frame_idx=np.concatenate(fidx_all) if fidx_all else np.zeros(0, np.int32),
            **inventory,
        ))
        n_det = int(sum(len(c) for c in conf_all))
        print(f'[INFO] кэш {i}/{len(video_ids)}: {vid} '
              f'({len(frames)} кадров, {n_det} детекций, {time.time()-t0:.1f}s)',
              flush=True)
    return out_dir


def load_video_cache(cache_dir: str | Path, split: str, video_id: str) -> dict:
    """Читает npz одного видео. Возвращает dict с массивами кэша."""
    return load_detection_arrays(Path(cache_dir) / split / f'{video_id}.npz')


def iter_frames(cache: dict):
    """Генератор по кадрам кэша: (frame_index, frame_name, xyxy, conf, cls).

    Кадры без детекций тоже выдаются (с пустыми массивами) — трекеру
    нужен каждый кадр для предсказания Калмана и учёта track_buffer.
    """
    frame_names = cache['frame_names']
    fidx = cache['frame_idx']
    for f in range(len(frame_names)):
        m = fidx == f
        yield (f, str(frame_names[f]), cache['xyxy'][m], cache['conf'][m],
               cache['cls'][m])
