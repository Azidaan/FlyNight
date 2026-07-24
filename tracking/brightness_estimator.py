"""Средняя яркость видео (Этап 8, подэтап 8.3).

Яркость = среднее V-канала HSV по пикселям, усреднённое по кадрам видео,
нормированное в [0, 1]. Для скорости сэмплируется каждый k-й кадр
(FRAME_STRIDE): освещение внутри 40-секундного ролика BDD меняется слабо.

Результаты кэшируются в experiments/results/brightness_per_video.csv
(колонки split, video_id, brightness) — повторные вызовы читают кэш.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402

CSV_PATH = PROJECT_ROOT / 'experiments' / 'results' / 'brightness_per_video.csv'
FRAME_STRIDE = 10


def compute_video_brightness(video_id: str, split: str = 'val') -> float:
    """Средняя яркость одного видео в [0, 1] (V-канал HSV)."""
    frames_dir = DATA_ROOT / 'images' / 'track' / split / video_id
    frames = sorted(frames_dir.glob('*.jpg'))[::FRAME_STRIDE]
    if not frames:
        raise FileNotFoundError(f'Нет кадров: {frames_dir}')
    vals = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        vals.append(float(hsv[:, :, 2].mean()))
    return float(np.mean(vals)) / 255.0


def load_brightness_cache() -> dict[tuple[str, str], float]:
    """Читает кэш яркостей: (split, video_id) -> brightness."""
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        return {(r['split'], r['video_id']): float(r['brightness'])
                for r in csv.DictReader(f)}


def get_brightness_map(video_ids: list[str], split: str = 'val') -> dict[str, float]:
    """Яркости списка видео; недостающие считаются и дописываются в кэш."""
    cache = load_brightness_cache()
    missing = [v for v in video_ids if (split, v) not in cache]
    if missing:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not CSV_PATH.exists()
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['split', 'video_id', 'brightness'])
            for i, vid in enumerate(missing, 1):
                b = compute_video_brightness(vid, split)
                cache[(split, vid)] = b
                w.writerow([split, vid, f'{b:.6f}'])
                if i % 25 == 0 or i == len(missing):
                    print(f'[INFO] яркость {i}/{len(missing)}', flush=True)
    return {v: cache[(split, v)] for v in video_ids}
