"""Визуальная проверка аугментаций перед финальным обучением (Этап 6, п. 6.2).

Строит train-датасет Ultralytics с параметрами аугментаций из
drb_aug_full.yaml и сохраняет мозаики нескольких батчей в figs/aug_examples/.
Проверяется глазами: рамки на местах, кадры не «убиты» (нет полностью
чёрных), Mosaic-сетки выглядят разумно.

Использование:
    python scripts/visualize_aug_batches.py
    python scripts/visualize_aug_batches.py --n-batches 10 --config training/configs/drb_aug_full.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_drb import AUG_KEYS  # noqa: E402
from training.train_baseline import ensure_data_yaml  # noqa: E402

FIGS = PROJECT_ROOT / 'figs' / 'aug_examples'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(
        PROJECT_ROOT / 'training' / 'configs' / 'drb_aug_full.yaml'))
    ap.add_argument('--n-batches', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.utils.plotting import plot_images

    cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    data_yaml_path = ensure_data_yaml(cfg_yaml)
    data = yaml.safe_load(data_yaml_path.read_text(encoding='utf-8'))
    data['channels'] = data.get('channels', 3)

    overrides = {'imgsz': cfg_yaml['imgsz'], 'task': 'detect', 'mode': 'train'}
    overrides.update({k: cfg_yaml[k] for k in AUG_KEYS if k in cfg_yaml})
    ucfg = get_cfg(overrides=overrides)

    train_txt = data['train']
    dataset = build_yolo_dataset(ucfg, train_txt, args.batch, data,
                                 mode='train')
    loader = build_dataloader(dataset, batch=args.batch, workers=0,
                              shuffle=True)

    FIGS.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        fname = FIGS / f'aug_batch_{i}.jpg'
        plot_images(labels=batch, fname=fname, save=True)
        imgs = batch['img']
        dark = float((imgs.float().mean(dim=(1, 2, 3)) < 5).sum())
        print(f'[OK ] {fname.name}: {tuple(imgs.shape)}, '
              f'полностью тёмных изображений: {dark:.0f}')
    print(f'[OK ] мозаики в {FIGS} — проверить глазами перед запуском '
          f'run_drb_aug_all_seeds.ps1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
