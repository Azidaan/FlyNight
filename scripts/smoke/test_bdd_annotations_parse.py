"""Smoke-тест парсинга аннотаций BDD100K (Этап 1, блок 4).

Две проверки:
1. Один MOT-файл labels/box_track_20/val/*.json парсится: извлекаются
   videoName, кадры, рамки box2d, категории. Подтверждаем, что тэгов
   weather/timeofday в MOT-аннотациях НЕТ (важное уточнение Этапа 2 плана).
2. Per-image Det-лейблы labels/bdd100k_labels_images_val.json загружаются,
   для каждого MOT-видео val-сплита находятся атрибуты weather/timeofday
   по ключу video_id == splitext(name)[0]. Ожидается покрытие 200/200.
   Дополнительно печатаются счётчики стресс-поднаборов по определениям
   из Части 2.9 плана (Normal/Night/Rain/Fog).
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / 'data' / 'bdd100k'


def parse_one_mot_json(path: Path) -> bool:
    with open(path, encoding='utf-8') as f:
        frames = json.load(f)
    if not isinstance(frames, list) or not frames:
        print(f'[FAIL] {path.name}: ожидался непустой список кадров')
        return False

    first = frames[0]
    video_name = first.get('videoName', '<нет>')
    n_frames = len(frames)
    n_labels = sum(len(fr.get('labels') or []) for fr in frames)
    categories = Counter(
        lb.get('category') for fr in frames for lb in (fr.get('labels') or [])
    )
    sample_box = None
    for fr in frames:
        for lb in fr.get('labels') or []:
            if lb.get('box2d'):
                sample_box = lb['box2d']
                break
        if sample_box:
            break

    print(f'[OK ] MOT JSON: {path.name}')
    print(f'      videoName={video_name}, кадров={n_frames}, рамок={n_labels}')
    print(f'      категории: {dict(categories)}')
    print(f'      пример box2d: {sample_box}')

    # В MOT-аннотациях НЕ должно быть weather/timeofday — тэги живут в Det-лейблах.
    frame_attr_keys = set()
    for fr in frames:
        frame_attr_keys.update((fr.get('attributes') or {}).keys())
    has_tags = {'weather', 'timeofday'} & frame_attr_keys
    if has_tags:
        print(f'[WARN] Неожиданно найдены тэги в MOT JSON: {has_tags} — план предполагает обратное')
    else:
        print('[OK ] weather/timeofday в MOT JSON отсутствуют — источник тэгов Det-лейблы, как в плане')

    return sample_box is not None and n_frames > 0


def check_det_labels_coverage() -> bool:
    det_path = DATA / 'labels' / 'bdd100k_labels_images_val.json'
    mot_val_dir = DATA / 'images' / 'track' / 'val'
    mot_video_ids = sorted(p.name for p in mot_val_dir.iterdir() if p.is_dir())
    print(f'[INFO] MOT val видео: {len(mot_video_ids)}')

    print(f'[INFO] Загрузка {det_path.name} (~200 МБ)...')
    with open(det_path, encoding='utf-8') as f:
        entries = json.load(f)
    print(f'[INFO] Записей в Det val: {len(entries)}')

    video_attrs = {}
    for entry in entries:
        vid = os.path.splitext(entry.get('name', ''))[0]
        video_attrs[vid] = entry.get('attributes') or {}

    covered = [v for v in mot_video_ids if v in video_attrs]
    missing = [v for v in mot_video_ids if v not in video_attrs]
    print(f'[{"OK " if not missing else "FAIL"}] Покрытие MOT-видео атрибутами: '
          f'{len(covered)}/{len(mot_video_ids)}')
    if missing:
        print(f'      без атрибутов: {missing[:10]}{"..." if len(missing) > 10 else ""}')

    # Счётчики стресс-поднаборов по определениям Части 2.9 плана.
    subsets = {
        'normal': lambda a: a.get('timeofday') == 'daytime' and a.get('weather') == 'clear',
        'night': lambda a: a.get('timeofday') == 'night',
        'rain': lambda a: a.get('weather') == 'rainy',
        'fog': lambda a: a.get('weather') in ('foggy', 'overcast'),
    }
    for name, pred in subsets.items():
        n = sum(1 for v in covered if pred(video_attrs[v]))
        print(f'      поднабор {name:<7}: {n} видео (val)')

    weather = Counter(video_attrs[v].get('weather') for v in covered)
    timeofday = Counter(video_attrs[v].get('timeofday') for v in covered)
    print(f'      weather   : {dict(weather)}')
    print(f'      timeofday : {dict(timeofday)}')

    return not missing


def main() -> int:
    mot_dir = DATA / 'labels' / 'box_track_20' / 'val'
    mot_files = sorted(mot_dir.glob('*.json'))
    if not mot_files:
        print(f'[FAIL] Нет MOT JSON в {mot_dir}')
        return 1

    ok_mot = parse_one_mot_json(mot_files[0])
    ok_det = check_det_labels_coverage()

    if ok_mot and ok_det:
        print('\n--- OK: аннотации BDD100K парсятся, тэги извлекаются ---')
        return 0
    print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
    return 1


if __name__ == '__main__':
    sys.exit(main())
