"""Проверка фильтра поднаборов: счётчики видео и кадров (Этап 2).

Для обоих сплитов (train, val) и каждого поднабора (normal/night/rain/fog)
выводит количество видео и кадров. Числа фиксируются в notes/subset_counts.json
и используются в разделе 2.3 диссертации.

Предупреждает, если train-поднабор меньше 50 видео (критерий Части 2.9 плана —
кандидат на расширение определения).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.subset_filter import (  # noqa: E402
    SUBSET_DEFINITIONS, filter_videos_by_subset, load_video_attributes,
)

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
OUT_JSON = PROJECT_ROOT / 'notes' / 'subset_counts.json'
MIN_TRAIN_VIDEOS = 50


def count_frames(split: str, video_ids: list[str]) -> int:
    img_root = DATA / 'images' / 'track' / split
    return sum(
        sum(1 for f in (img_root / vid).glob('*.jpg')) for vid in video_ids
    )


def main() -> int:
    report: dict = {}
    warnings: list[str] = []

    for split in ('train', 'val'):
        det_json = DATA / 'labels' / f'bdd100k_labels_images_{split}.json'
        print(f'[INFO] Загрузка атрибутов {det_json.name}...', flush=True)
        video_attrs = load_video_attributes(str(det_json))

        img_root = DATA / 'images' / 'track' / split
        mot_video_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())
        covered = sum(1 for v in mot_video_ids if v in video_attrs)
        print(f'[INFO] {split}: MOT-видео {len(mot_video_ids)}, '
              f'с атрибутами {covered}')

        report[split] = {
            'total_videos': len(mot_video_ids),
            'with_attributes': covered,
            'subsets': {},
        }
        for subset in SUBSET_DEFINITIONS:
            vids = filter_videos_by_subset(video_attrs, mot_video_ids, subset)
            n_frames = count_frames(split, vids)
            report[split]['subsets'][subset] = {
                'videos': len(vids), 'frames': n_frames,
            }
            print(f'  {subset:<7}: {len(vids):>4} видео, {n_frames:>6} кадров')
            if split == 'train' and len(vids) < MIN_TRAIN_VIDEOS:
                warnings.append(
                    f'поднабор {subset} в train: {len(vids)} видео '
                    f'< {MIN_TRAIN_VIDEOS} — кандидат на расширение определения'
                )

    report['definitions'] = SUBSET_DEFINITIONS
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\n[INFO] Счётчики сохранены: {OUT_JSON}')

    for w in warnings:
        print(f'[WARN] {w}')
    if all(
        report[s]['subsets'][sub]['videos'] > 0
        for s in ('train', 'val') for sub in SUBSET_DEFINITIONS
    ):
        print('\n--- OK: все поднаборы непусты в обоих сплитах ---')
        return 0
    print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА: есть пустой поднабор ---')
    return 1


if __name__ == '__main__':
    sys.exit(main())
