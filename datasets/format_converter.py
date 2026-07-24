"""Конвертация аннотаций BDD100K MOT (box_track_20) в YOLO-формат (Этап 2).

Один MOT JSON (список кадров одного видео) -> по одному txt-файлу на кадр:
строки вида `class cx cy w h` с нормализованными к [0, 1] координатами.

Договорённости (фиксируются здесь и в notes/subset_definitions.md):
- Классы: 8 категорий BDD100K MOT (в Det-таксономии их 10, но в box_track_20
  traffic light / traffic sign отсутствуют — обучение идёт на MOT-кадрах).
- Рамки с attributes.crowd == True пропускаются (в официальном протоколе BDD
  это ignore-регионы), их количество логируется.
- Неизвестные категории (например, 'other person', 'other vehicle', 'trailer')
  пропускаются с подсчётом.
- Кадры без валидных рамок получают пустой txt (фоновые изображения для YOLO).
- Размер кадра BDD100K MOT фиксирован: 1280x720 (проверяется вызывающим кодом).
"""
import json
from pathlib import Path

IMG_W, IMG_H = 1280, 720

# Официальный порядок категорий BDD100K MOT.
CLASS_MAPPING: dict[str, int] = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
}


def convert_bdd_to_yolo(
    bdd_json_path: str,
    yolo_labels_dir: str,
    class_mapping: dict[str, int] = CLASS_MAPPING,
) -> dict:
    """Конвертирует один MOT JSON (одно видео) в папку YOLO txt-файлов.

    Возвращает статистику: {'frames': int, 'boxes': int, 'crowd_skipped': int,
    'unknown_skipped': dict[str, int]}.
    """
    out_dir = Path(yolo_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(bdd_json_path, encoding='utf-8') as f:
        frames = json.load(f)

    stats = {'frames': 0, 'boxes': 0, 'crowd_skipped': 0, 'unknown_skipped': {}}
    for frame in frames:
        lines = []
        for label in frame.get('labels') or []:
            box = label.get('box2d')
            if not box:
                continue
            if (label.get('attributes') or {}).get('crowd'):
                stats['crowd_skipped'] += 1
                continue
            category = label.get('category')
            if category not in class_mapping:
                stats['unknown_skipped'][category] = \
                    stats['unknown_skipped'].get(category, 0) + 1
                continue
            x1 = min(max(box['x1'], 0.0), IMG_W)
            x2 = min(max(box['x2'], 0.0), IMG_W)
            y1 = min(max(box['y1'], 0.0), IMG_H)
            y2 = min(max(box['y2'], 0.0), IMG_H)
            w, h = x2 - x1, y2 - y1
            if w <= 1.0 or h <= 1.0:  # вырожденная рамка после клампа
                continue
            cx, cy = (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H
            lines.append(
                f"{class_mapping[category]} {cx:.6f} {cy:.6f} "
                f"{w / IMG_W:.6f} {h / IMG_H:.6f}"
            )
            stats['boxes'] += 1
        frame_stem = Path(frame['name']).stem
        (out_dir / f'{frame_stem}.txt').write_text(
            '\n'.join(lines), encoding='utf-8'
        )
        stats['frames'] += 1
    return stats


def convert_split(
    mot_labels_dir: str,
    yolo_labels_root: str,
    class_mapping: dict[str, int] = CLASS_MAPPING,
    log_every: int = 200,
) -> dict:
    """Конвертирует весь split: labels/box_track_20/<split>/ -> <yolo_root>/<video>/.

    Возвращает агрегированную статистику по всем видео.
    """
    mot_dir = Path(mot_labels_dir)
    json_files = sorted(mot_dir.glob('*.json'))
    total = {'videos': 0, 'frames': 0, 'boxes': 0, 'crowd_skipped': 0,
             'unknown_skipped': {}}
    for i, jf in enumerate(json_files, 1):
        stats = convert_bdd_to_yolo(
            str(jf), str(Path(yolo_labels_root) / jf.stem), class_mapping
        )
        total['videos'] += 1
        total['frames'] += stats['frames']
        total['boxes'] += stats['boxes']
        total['crowd_skipped'] += stats['crowd_skipped']
        for cat, n in stats['unknown_skipped'].items():
            total['unknown_skipped'][cat] = total['unknown_skipped'].get(cat, 0) + n
        if i % log_every == 0 or i == len(json_files):
            print(f'  [{i}/{len(json_files)}] кадров={total["frames"]}, '
                  f'рамок={total["boxes"]}', flush=True)
    return total
