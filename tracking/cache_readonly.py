"""Явное чтение существующего кэша без достройки и присвоения происхождения."""
from pathlib import Path


def require_cached_videos(cache_dir, split, video_ids, *, required=()):
    """Проверить все выбранные видео до записи результатов анализа.

    Старые кэши допускаются, но наличие файлов не подтверждает происхождение
    весов. Эта функция не принимает модель, не создаёт кэш и не меняет meta.json.
    """
    from tracking.cache_integrity import validate_detection_arrays
    from tracking.detection_cache import load_video_cache

    video_ids = list(video_ids)
    if not video_ids:
        raise ValueError('Не выбрано ни одного видео для чтения кэша')
    folder = Path(cache_dir) / split
    missing = [vid for vid in video_ids if not (folder / f'{vid}.npz').is_file()]
    if missing:
        raise FileNotFoundError(
            f'--cached-only: неполный кэш {folder}; отсутствуют видео {missing}. '
            'Укажите полный существующий кэш или создайте новый в другом каталоге.')
    for vid in video_ids:
        try:
            cache = load_video_cache(cache_dir, split, vid)
            validate_detection_arrays(cache, required=required)
        except (ValueError, KeyError) as error:
            raise ValueError(f'--cached-only: некорректный кэш {vid}: {error}') from error
    print(f'[CACHE] Проверены массивы {len(video_ids)} видео; исходные веса '
          'в режиме --cached-only не проверяются. Мета кэша не изменена.')
    return folder
