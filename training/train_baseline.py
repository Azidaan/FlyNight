"""Обучение baseline YOLO без модификаций (Этап 3).

Один запуск = один seed. Три сида (42, 123, 2024) запускаются
scripts/run_baseline_all_seeds.ps1 последовательно.

Использование:
    python training/train_baseline.py --seed 42
    python training/train_baseline.py --seed 42 --resume   # продолжить прерванное

Выходы (для seed 42):
    runs/baseline/seed42/                — логи Ultralytics, чекпойнты, results.csv
    runs/baseline/seed42/eval_metrics_epoch_*.json — метрики блоков по 10 эпох
    weights/baseline_seed42/{best,last}.pt         — финальные веса (копии)
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import (  # noqa: E402
    DATA_ROOT, build_subset_yaml, list_subset_frames,
)
from training.trainer_utils import (  # noqa: E402
    HeartbeatLogger, check_checkpoint_compatibility, setup_compute_flags,
)

NUM_CLASSES = 8


def ensure_data_yaml(cfg: dict) -> Path:
    """Генерирует (один раз) списки кадров полного датасета и data-yaml."""
    gen_dir = PROJECT_ROOT / 'training' / 'configs' / 'generated'
    yaml_path = gen_dir / f"{cfg['data_name']}.yaml"
    if yaml_path.exists():
        return yaml_path
    print('[INFO] Генерация списков кадров полного датасета '
          f'(train step {cfg["frame_step_train"]}, val step {cfg["frame_step_val"]})...',
          flush=True)
    train_ids = sorted(p.name for p in (DATA_ROOT / 'images' / 'track' / 'train').iterdir()
                       if p.is_dir())
    val_ids = sorted(p.name for p in (DATA_ROOT / 'images' / 'track' / 'val').iterdir()
                     if p.is_dir())
    train_frames = list_subset_frames('train', train_ids, cfg['frame_step_train'])
    val_frames = list_subset_frames('val', val_ids, cfg['frame_step_val'])
    print(f'[INFO] Кадров: train {len(train_frames)}, val {len(val_frames)}')
    return build_subset_yaml(cfg['data_name'], train_frames, val_frames)


def train_baseline(config_path: str, seed: int, output_dir: str,
                   resume: bool = False, data_override: str | None = None) -> str:
    """Запускает обучение одного сида. Возвращает путь к best.pt."""
    setup_compute_flags()
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    # только абсолютный путь: относительный Ultralytics разрешает от своего
    # каталога (runs/detect), и результаты уезжают мимо панели и супервизора
    out_root = Path(output_dir)
    if not out_root.is_absolute():
        out_root = (PROJECT_ROOT / out_root).resolve()
    run_dir = out_root / f'seed{seed}'
    # Имя берётся из фактического каталога прогона, а не зашито. Иначе
    # короткий контрольный прогон, запущенный с --output, затирает веса
    # настоящего baseline — а на них опираются оценка по поднаборам,
    # трекинг и замер скорости.
    weights_out = PROJECT_ROOT / 'weights' / f'{out_root.name}_seed{seed}'

    data_yaml = Path(data_override) if data_override else ensure_data_yaml(cfg)

    last = run_dir / 'weights' / 'last.pt'
    if resume and not last.exists():
        print(f'[WARN] --resume указан, но {last} нет — начинаю с нуля')
        resume = False
    start_weights = str(last) if resume else str(PROJECT_ROOT / cfg['weights'])
    if not check_checkpoint_compatibility(start_weights,
                                          NUM_CLASSES if resume else 80):
        # предобученные веса COCO имеют 80 классов — это норма для старта;
        # проверка на 8 классов актуальна только при resume
        raise SystemExit(1)

    model = YOLO(start_weights)
    hb = HeartbeatLogger(run_dir, block_epochs=cfg['block_epochs'])
    hb.attach(model)

    model.train(
        data=str(data_yaml), epochs=cfg['epochs'], batch=cfg['batch'],
        imgsz=cfg['imgsz'], workers=cfg['workers'], seed=seed, device=0,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        resume=resume, plots=True, verbose=True,
    )

    weights_out.mkdir(parents=True, exist_ok=True)
    best_src = run_dir / 'weights' / 'best.pt'
    for name in ('best.pt', 'last.pt'):
        src = run_dir / 'weights' / name
        if src.exists():
            shutil.copy2(src, weights_out / name)
    print(f'[OK ] Сид {seed}: веса скопированы в {weights_out}')
    return str(weights_out / 'best.pt') if best_src.exists() else ''


def main() -> int:
    ap = argparse.ArgumentParser(description='Обучение baseline YOLO (Этап 3)')
    ap.add_argument('--seed', type=int, required=True, choices=[42, 123, 2024])
    ap.add_argument('--config', default=str(PROJECT_ROOT / 'training' / 'configs' / 'baseline.yaml'))
    ap.add_argument('--output', default=str(PROJECT_ROOT / 'runs' / 'baseline'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--epochs', type=int, default=None,
                    help='переопределить число эпох (для smoke-проверки обвязки)')
    ap.add_argument('--data', default=None,
                    help='переопределить data-yaml (smoke на mini-BDD)')
    args = ap.parse_args()

    if args.epochs is not None:  # smoke-режим: короткий прогон
        cfg_path = Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        cfg['epochs'] = args.epochs
        tmp = cfg_path.parent / 'generated' / f'baseline_e{args.epochs}.yaml'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
        args.config = str(tmp)

    best = train_baseline(args.config, args.seed, args.output, args.resume,
                          data_override=args.data)
    print(f'[OK ] best.pt: {best}')
    return 0 if best else 1


if __name__ == '__main__':
    sys.exit(main())
