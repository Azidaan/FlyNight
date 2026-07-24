"""Фильтрация видео BDD100K MOT по стресс-поднаборам (Этап 2, подэтап 2.3).

Источник тэгов weather/timeofday — per-image Det-лейблы
(labels/bdd100k_labels_images_{train,val}.json), НЕ MOT JSON: в box_track_20
атрибутов погоды нет. Сопоставление по ключу
video_id == os.path.splitext(entry['name'])[0].

Определения поднаборов зафиксированы в notes/subset_definitions.md и в Части 2.9
плана. Fog сразу расширен до {foggy, overcast}: видео с тэгом foggy в BDD100K
единицы (1 train / 0 val), узкое определение статистически невалидно.

Фильтр не копирует данные — возвращает список video_id.
"""
import json
import os

# Единственное место, где определены поднаборы. При изменении — пересчитать
# все эксперименты и обновить notes/subset_definitions.md.
SUBSET_DEFINITIONS: dict[str, str] = {
    'normal': "timeofday == 'daytime' и weather == 'clear'",
    'night': "timeofday == 'night' (любая погода)",
    'rain': "weather == 'rainy' (любое время суток)",
    'fog': "weather in {'foggy', 'overcast'} (любое время суток)",
}

_PREDICATES = {
    'normal': lambda a: a.get('timeofday') == 'daytime' and a.get('weather') == 'clear',
    'night': lambda a: a.get('timeofday') == 'night',
    'rain': lambda a: a.get('weather') == 'rainy',
    'fog': lambda a: a.get('weather') in ('foggy', 'overcast'),
}


def load_video_attributes(det_labels_json_path: str) -> dict[str, dict]:
    """Читает большой per-image JSON BDD100K Det и возвращает video_id -> attributes.

    attributes = {'weather': str, 'timeofday': str, 'scene': str}.
    Файл train ~1.4 ГБ, загрузка десятки секунд — вызывать один раз и
    переиспользовать результат.
    """
    with open(det_labels_json_path, encoding='utf-8') as f:
        entries = json.load(f)
    video_attrs: dict[str, dict] = {}
    for entry in entries:
        video_id = os.path.splitext(entry.get('name', ''))[0]
        video_attrs[video_id] = entry.get('attributes') or {}
    return video_attrs


def filter_videos_by_subset(
    video_attrs: dict[str, dict],
    mot_video_ids: list[str],
    subset_name: str,
) -> list[str]:
    """Пересекает MOT video IDs с попадающими в поднабор по атрибутам.

    mot_video_ids — имена директорий в images/track/<split>/.
    subset_name — 'normal' | 'night' | 'rain' | 'fog'.
    Видео без атрибутов (нет в Det-лейблах) в поднаборы не включаются.
    """
    if subset_name not in _PREDICATES:
        raise ValueError(
            f'Неизвестный поднабор {subset_name!r}; допустимы {sorted(_PREDICATES)}'
        )
    pred = _PREDICATES[subset_name]
    return [
        vid for vid in mot_video_ids
        if vid in video_attrs and pred(video_attrs[vid])
    ]
