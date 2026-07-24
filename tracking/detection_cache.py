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
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402

CACHE_ROOT = PROJECT_ROOT / 'runs' / 'det_cache'
CACHE_CONF = 0.01   # NMS confidence при кэшировании (ниже любого порога сетки)
CACHE_IOU = 0.7     # NMS IoU — дефолт Ultralytics predict


def build_detection_cache(
    model_path: str,
    video_ids: list[str],
    split: str = 'val',
    cache_dir: str | Path = CACHE_ROOT,
    device: str | int = 0,
) -> Path:
    """Прогоняет детектор по видео и пишет npz-кэш. Пропускает готовые видео.

    Возвращает директорию кэша для split.
    """
    from ultralytics import YOLO

    cache_dir = Path(cache_dir)
    out_dir = cache_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = DATA_ROOT / 'images' / 'track' / split

    meta_path = cache_dir / 'meta.json'
    meta = {'model': str(model_path), 'conf': CACHE_CONF, 'iou': CACHE_IOU,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding='utf-8'))
        if old.get('model') != meta['model']:
            raise RuntimeError(
                f'Кэш уже построен другой моделью ({old.get("model")}). '
                f'Удалите {cache_dir} или укажите другую директорию.')
    else:
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    model = YOLO(model_path)
    for i, vid in enumerate(video_ids, 1):
        out_npz = out_dir / f'{vid}.npz'
        if out_npz.exists():
            continue
        frames = sorted((img_root / vid).glob('*.jpg'))
        xyxy_all, conf_all, cls_all, fidx_all = [], [], [], []
        t0 = time.time()
        results = model.predict(source=str(img_root / vid), conf=CACHE_CONF,
                                iou=CACHE_IOU, device=device, stream=True,
                                verbose=False)
        for fidx, res in enumerate(results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            n = len(boxes)
            xyxy_all.append(boxes.xyxy.cpu().numpy().astype(np.float32))
            conf_all.append(boxes.conf.cpu().numpy().astype(np.float32))
            cls_all.append(boxes.cls.cpu().numpy().astype(np.int16))
            fidx_all.append(np.full(n, fidx, dtype=np.int32))
        np.savez_compressed(
            out_npz,
            xyxy=np.concatenate(xyxy_all) if xyxy_all else np.zeros((0, 4), np.float32),
            conf=np.concatenate(conf_all) if conf_all else np.zeros(0, np.float32),
            cls=np.concatenate(cls_all) if cls_all else np.zeros(0, np.int16),
            frame_idx=np.concatenate(fidx_all) if fidx_all else np.zeros(0, np.int32),
            frame_names=np.array([f.name for f in frames]),
        )
        n_det = int(sum(len(c) for c in conf_all))
        print(f'[INFO] кэш {i}/{len(video_ids)}: {vid} '
              f'({len(frames)} кадров, {n_det} детекций, {time.time()-t0:.1f}s)',
              flush=True)
    return out_dir


def load_video_cache(cache_dir: str | Path, split: str, video_id: str) -> dict:
    """Читает npz одного видео. Возвращает dict с массивами кэша."""
    npz = np.load(Path(cache_dir) / split / f'{video_id}.npz')
    return {k: npz[k] for k in npz.files}


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
