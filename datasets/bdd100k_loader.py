"""Тонкий загрузчик BDD100K MOT для Ultralytics (Этап 2, подэтап 2.3).

Ultralytics принимает в data-yaml вместо папки txt-файл со списком путей
к изображениям. Здесь собирается такой список для выбранного поднабора
(с опциональным прореживанием кадров, Часть 2.9 плана: каждый 5-й кадр
для обучения детектора) и генерируется yaml.

Пути к разметке Ultralytics выводит сам заменой /images/ -> /labels/:
  data/bdd100k/images/track/train/<video>/<frame>.jpg
  data/bdd100k/labels/track/train/<video>/<frame>.txt  (создаёт format_converter)

Дополнительно — защита от зависания DataLoader на Windows в debug-режиме.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data' / 'bdd100k'

CLASS_NAMES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle',
}


def detect_debug_mode() -> bool:
    """Debug-режим: активный трейсер (debugpy/pdb) или переменная окружения."""
    return sys.gettrace() is not None or os.environ.get('FORCE_DEBUG_MODE') == '1'


def get_dataloader_kwargs(is_debug: bool | None = None) -> dict:
    """Безопасные параметры DataLoader для Windows.

    В debug: воркеры отключены (DataLoader с num_workers>0 в debug зависает).
    В норме: 4 воркера с persistent_workers.
    """
    if is_debug is None:
        is_debug = detect_debug_mode()
    if is_debug:
        return {'num_workers': 0, 'persistent_workers': False}
    return {'num_workers': 4, 'persistent_workers': True}


def list_subset_frames(
    split: str,
    video_ids: list[str],
    frame_step: int = 1,
    data_root: Path = DATA_ROOT,
) -> list[Path]:
    """Список путей к кадрам указанных видео с прореживанием frame_step.

    Кадры внутри видео сортируются по имени (совпадает с порядком по времени);
    берётся каждый frame_step-й, начиная с первого. frame_step одинаков для
    всех конфигураций эксперимента (честность сравнений, Часть 2.9).
    """
    frames: list[Path] = []
    img_root = data_root / 'images' / 'track' / split
    for vid in video_ids:
        video_dir = img_root / vid
        if not video_dir.is_dir():
            raise FileNotFoundError(f'Нет папки кадров: {video_dir}')
        jpgs = sorted(video_dir.glob('*.jpg'))
        frames.extend(jpgs[::frame_step])
    return frames


def build_subset_yaml(
    name: str,
    train_frames: list[Path],
    val_frames: list[Path],
    out_dir: Path | str = PROJECT_ROOT / 'training' / 'configs' / 'generated',
) -> Path:
    """Пишет списки кадров (txt) и data-yaml для Ultralytics. Возвращает путь к yaml.

    name — идентификатор конфигурации (например, 'full_step5' или 'mini_bdd').
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_txt = out_dir / f'{name}_train.txt'
    val_txt = out_dir / f'{name}_val.txt'
    train_txt.write_text(
        '\n'.join(str(p) for p in train_frames), encoding='utf-8'
    )
    val_txt.write_text('\n'.join(str(p) for p in val_frames), encoding='utf-8')

    names_block = '\n'.join(f'  {i}: {n}' for i, n in CLASS_NAMES.items())
    yaml_path = out_dir / f'{name}.yaml'
    yaml_path.write_text(
        f'# Автогенерация: datasets/bdd100k_loader.py, конфигурация {name}\n'
        f'train: {train_txt.as_posix()}\n'
        f'val: {val_txt.as_posix()}\n'
        f'names:\n{names_block}\n',
        encoding='utf-8',
    )
    return yaml_path
