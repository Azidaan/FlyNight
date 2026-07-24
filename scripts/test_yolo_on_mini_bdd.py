"""Проверка совместимости пайплайна данных с YOLO26 (Этап 2, критерий готовности).

Короткое дообучение YOLO26s на mini-BDD через yaml, сгенерированный
datasets/bdd100k_loader.build_subset_yaml. Проверяется вся цепочка:
кадры BDD -> YOLO txt (format_converter) -> список путей -> data-yaml ->
Ultralytics train/val. Это НЕ эксперимент для диссертации, только smoke.

Разбиение mini-BDD: по одному видео каждого условия в val, остальные в train.
Прореживание: train — каждый 5-й кадр (как в будущих обучениях), val — каждый 10-й.

Критерий: обучение завершилось без ошибок, mAP@0.50 на val > 0.05
(предобученная на COCO модель за пару эпох обязана давать заметный mAP
на дорожных классах).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import build_subset_yaml  # noqa: E402

MINI = PROJECT_ROOT / 'data' / 'bdd100k_mot_mini'
WEIGHTS = PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt'
RUNS = PROJECT_ROOT / 'runs' / 'smoke'


def split_videos() -> tuple[list[str], list[str]]:
    manifest = json.loads((MINI / 'manifest.json').read_text(encoding='utf-8'))
    val_ids, train_ids, seen_subsets = [], [], set()
    for m in manifest:
        if m['subset'] not in seen_subsets:
            seen_subsets.add(m['subset'])
            val_ids.append(m['video_id'])
        else:
            train_ids.append(m['video_id'])
    return train_ids, val_ids


def list_frames(video_ids: list[str], step: int) -> list[Path]:
    frames: list[Path] = []
    for vid in video_ids:
        jpgs = sorted((MINI / 'images' / vid).glob('*.jpg'))
        frames.extend(jpgs[::step])
    return frames


def main() -> int:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    from ultralytics import YOLO

    train_ids, val_ids = split_videos()
    train_frames = list_frames(train_ids, step=5)
    val_frames = list_frames(val_ids, step=10)
    print(f'[INFO] mini-BDD: train {len(train_ids)} видео / {len(train_frames)} кадров, '
          f'val {len(val_ids)} видео / {len(val_frames)} кадров')

    yaml_path = build_subset_yaml('mini_bdd_smoke', train_frames, val_frames)
    print(f'[INFO] data-yaml: {yaml_path}')

    model = YOLO(str(WEIGHTS))
    results = model.train(
        data=str(yaml_path), epochs=3, imgsz=640, batch=8, workers=4,
        device=0, seed=42, plots=False, project=str(RUNS),
        name='mini_bdd_train', exist_ok=True, verbose=True,
    )
    map50 = float(results.box.map50)
    print(f'\n[INFO] mAP@0.50 на mini-BDD val: {map50:.4f}')

    if map50 > 0.05:
        print('--- OK: пайплайн данных совместим с YOLO26 ---')
        return 0
    print('--- ПРОВЕРКА НЕ ПРОЙДЕНА: mAP подозрительно низкий, проверить разметку ---')
    return 1


if __name__ == '__main__':
    sys.exit(main())
