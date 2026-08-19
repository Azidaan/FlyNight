"""Обучение YOLO + DRB (Этап 5; варианты ветвей — Этап 7).

Один запуск = один сид. Конфигурация полностью повторяет baseline
(training/configs/drb_full.yaml), отличие — DRB, вставляемый DRBTrainer-ом.

Дополнительно к baseline: после каждой эпохи в <run_dir>/drb_alphas.csv
пишутся текущие α₁ = σ(w_LQA), α₂ = σ(w_FGA) по уровням пирамиды — динамика
обучаемых весов ветвей нужна для графиков Этапа 5 и анализа специализации.

Использование:
    python training/train_drb.py --seed 42
    python training/train_drb.py --seed 42 --variant lqa      # ablation Этапа 7
    python training/train_drb.py --seed 42 --variant input    # DRB на RGB-входе
    python training/train_drb.py --seed 42 --resume

Выходы (для seed 42, variant both):
    runs/drb_full/seed42/           — логи Ultralytics, чекпойнты, drb_alphas.csv
    weights/drb_full_seed42/{best,last}.pt
Для вариантов: runs/ablation_<variant>_only/seed42/, weights/ablation_<variant>_only_seed42/.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo_neck import iter_drb, make_drb_trainer  # noqa: E402
from training.train_baseline import ensure_data_yaml  # noqa: E402
from training.trainer_utils import (  # noqa: E402
    HeartbeatLogger, check_checkpoint_compatibility, setup_compute_flags,
)

NUM_CLASSES = 8

VARIANT_RUN_NAME = {
    'both': 'drb_full',
    'lqa': 'ablation_lqa_only',
    'fga': 'ablation_fga_only',
    'input': 'ablation_drb_input',
}

# Параметры аугментаций Ultralytics, пробрасываемые из yaml в model.train()
# (Этап 6; в конфигах без этих ключей — дефолты Ultralytics, как у baseline).
AUG_KEYS = ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
            'mosaic', 'mixup', 'cutmix', 'copy_paste', 'erasing',
            'auto_augment')


def make_alpha_logger(run_dir: Path):
    """Callback on_fit_epoch_end: дописывает α по уровням в drb_alphas.csv."""
    csv_path = run_dir / 'drb_alphas.csv'

    def _log(trainer) -> None:
        rows = []
        for idx, drb in iter_drb(trainer.model):
            a1, a2 = drb.alphas
            rows.append([trainer.epoch, idx, f'{a1:.6f}', f'{a2:.6f}'])
        if not rows:
            return
        write_header = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['epoch', 'layer_idx', 'alpha_lqa', 'alpha_fga'])
            w.writerows(rows)

    return _log


def train_drb(config_path: str, seed: int, output_dir: str | None = None,
              variant: str = 'both', resume: bool = False,
              data_override: str | None = None,
              alpha_init: float | None = None) -> str:
    """Обучение одного сида YOLO+DRB. Возвращает путь к best.pt."""
    setup_compute_flags()
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    variant = cfg.get('drb_variant', variant) if variant == 'both' else variant
    run_name = cfg.get('run_name') or VARIANT_RUN_NAME[variant]
    aug_kwargs = {k: cfg[k] for k in AUG_KEYS if k in cfg}
    if aug_kwargs:
        print(f'[INFO] аугментации из конфига: {aug_kwargs}')
    # Путь ОБЯЗАН быть абсолютным: относительный Ultralytics разрешает от
    # своего каталога (runs/detect), и результаты уезжают в
    # runs/detect/runs/... — панель их не видит, а супервизор не находит
    # чекпойнт в наблюдаемой папке и через грацию убивает обучение как зависшее.
    run_root = Path(output_dir) if output_dir else PROJECT_ROOT / 'runs' / run_name
    if not run_root.is_absolute():
        run_root = (PROJECT_ROOT / run_root).resolve()
    run_dir = run_root / f'seed{seed}'
    # Имя из фактического каталога прогона, а не из конфига: пробы идут с
    # --output в runs/init_probe/<вариант>, и по имени из конфига их
    # 12-эпошные веса ложились в weights/drb_full_seed42, то есть под
    # именем настоящего прогона.
    weights_out = PROJECT_ROOT / 'weights' / f'{run_root.name}_seed{seed}'

    data_yaml = Path(data_override) if data_override else ensure_data_yaml(cfg)

    last = run_dir / 'weights' / 'last.pt'
    if resume and not last.exists():
        print(f'[WARN] --resume указан, но {last} нет — начинаю с нуля')
        resume = False
    start_weights = str(last) if resume else str(PROJECT_ROOT / cfg['weights'])
    if not check_checkpoint_compatibility(start_weights,
                                          NUM_CLASSES if resume else 80):
        raise SystemExit(1)

    model = YOLO(start_weights)
    hb = HeartbeatLogger(run_dir, block_epochs=cfg['block_epochs'])
    hb.attach(model)
    model.add_callback('on_fit_epoch_end', make_alpha_logger(run_dir))

    w0 = cfg.get('alpha_init', 0.0) if alpha_init is None else alpha_init
    model.train(
        trainer=make_drb_trainer(variant, alpha_init=w0),
        data=str(data_yaml), epochs=cfg['epochs'], batch=cfg['batch'],
        imgsz=cfg['imgsz'], workers=cfg['workers'], seed=seed, device=0,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        resume=resume, plots=True, verbose=True, **aug_kwargs,
    )

    weights_out.mkdir(parents=True, exist_ok=True)
    best_src = run_dir / 'weights' / 'best.pt'
    for name in ('best.pt', 'last.pt'):
        src = run_dir / 'weights' / name
        if src.exists():
            shutil.copy2(src, weights_out / name)
    print(f'[OK ] Сид {seed} ({run_name}): веса скопированы в {weights_out}')
    return str(weights_out / 'best.pt') if best_src.exists() else ''


def main() -> int:
    ap = argparse.ArgumentParser(description='Обучение YOLO + DRB (Этапы 5, 7)')
    ap.add_argument('--seed', type=int, required=True, choices=[42, 123, 2024])
    ap.add_argument('--variant', default='both',
                    choices=['both', 'lqa', 'fga', 'input'])
    ap.add_argument('--config', default=str(
        PROJECT_ROOT / 'training' / 'configs' / 'drb_full.yaml'))
    ap.add_argument('--output', default=None,
                    help='переопределить каталог запусков')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--epochs', type=int, default=None,
                    help='переопределить число эпох (smoke или ablation 50)')
    ap.add_argument('--data', default=None,
                    help='переопределить data-yaml (smoke на mini-BDD)')
    ap.add_argument('--alpha-init', type=float, default=None,
                    help='стартовые веса ветвей w_LQA=w_FGA (0 — вклад 50%%, '
                         '-2 — 12%%, -4 — почти тождественный старт)')
    args = ap.parse_args()

    if args.epochs is not None:
        cfg_path = Path(args.config)
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        cfg['epochs'] = args.epochs
        tmp = cfg_path.parent / 'generated' / (
            f'{cfg_path.stem}_{args.variant}_e{args.epochs}.yaml')
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
        args.config = str(tmp)

    best = train_drb(args.config, args.seed, args.output, args.variant,
                     args.resume, data_override=args.data,
                     alpha_init=args.alpha_init)
    print(f'[OK ] best.pt: {best}')
    return 0 if best else 1


if __name__ == '__main__':
    sys.exit(main())
