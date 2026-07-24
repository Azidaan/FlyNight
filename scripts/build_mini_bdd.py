"""Сборка mini-BDD из локального полного датасета (Этап 2, подэтап 2.1).

20 train-видео с разнообразными условиями: 8 normal, 6 night, 4 rain, 2 fog.
Выбор детерминированный: сортированный список video_id каждого поднабора,
первые N (видео, попавшие в несколько поднаборов, не дублируются).

Структура (совместима с Ultralytics: /images/ <-> /labels/):
data/bdd100k_mot_mini/
├── images/<video>/*.jpg          # копии кадров
├── labels/<video>/*.txt          # YOLO-разметка (через format_converter)
├── annotations_json/<video>.json # оригинальные MOT JSON
└── manifest.json                 # какие видео выбраны и почему
"""
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.format_converter import convert_bdd_to_yolo  # noqa: E402
from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
MINI = PROJECT_ROOT / 'data' / 'bdd100k_mot_mini'
QUOTA = {'normal': 8, 'night': 6, 'rain': 4, 'fog': 2}


def main() -> int:
    det_json = DATA / 'labels' / 'bdd100k_labels_images_train.json'
    print('[INFO] Загрузка атрибутов train...', flush=True)
    video_attrs = load_video_attributes(str(det_json))

    img_root = DATA / 'images' / 'track' / 'train'
    mot_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())

    chosen: dict[str, str] = {}  # video_id -> поднабор
    for subset, quota in QUOTA.items():
        candidates = filter_videos_by_subset(video_attrs, mot_ids, subset)
        picked = 0
        for vid in candidates:
            if picked >= quota:
                break
            if vid in chosen:
                continue
            chosen[vid] = subset
            picked += 1
        if picked < quota:
            print(f'[WARN] {subset}: выбрано {picked}/{quota}')

    print(f'[INFO] Выбрано {len(chosen)} видео, копирование...')
    manifest = []
    for vid, subset in sorted(chosen.items()):
        src_img = img_root / vid
        dst_img = MINI / 'images' / vid
        if not dst_img.exists():
            shutil.copytree(src_img, dst_img)
        src_json = DATA / 'labels' / 'box_track_20' / 'train' / f'{vid}.json'
        dst_json_dir = MINI / 'annotations_json'
        dst_json_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_json, dst_json_dir / f'{vid}.json')
        stats = convert_bdd_to_yolo(str(src_json), str(MINI / 'labels' / vid))
        n_frames = sum(1 for _ in dst_img.glob('*.jpg'))
        manifest.append({
            'video_id': vid, 'subset': subset,
            'attributes': video_attrs.get(vid, {}),
            'frames': n_frames, 'boxes': stats['boxes'],
        })
        print(f'  [{subset:<7}] {vid}: {n_frames} кадров, {stats["boxes"]} рамок')

    (MINI / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    subsets_present = {m['subset'] for m in manifest}
    total_frames = sum(m['frames'] for m in manifest)
    print(f'\n[INFO] mini-BDD: {len(manifest)} видео, {total_frames} кадров, '
          f'условия: {sorted(subsets_present)}')
    if len(manifest) == sum(QUOTA.values()) and len(subsets_present) >= 3:
        print('--- OK: mini-BDD собран (>=3 типов условий) ---')
        return 0
    print('--- ПРОВЕРКА НЕ ПРОЙДЕНА ---')
    return 1


if __name__ == '__main__':
    sys.exit(main())
