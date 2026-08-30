"""Обучение YOLO с ФИКСИРОВАННЫМ низкочастотным фильтром в операторах шеи.

Зачем отдельный скрипт, а не флаг train_drb.py. У этой конструкции нет ни
одного обучаемого числа: ни весов ветвей, ни маски, ни силы вмешательства.
Значит не нужно ничего из того, ради чего существует DRBTrainer — ни своих
групп оптимизатора, ни вывода параметров из-под штрафа за величину, ни
разгона скорости обучения блока, ни журнала альф. Тренер здесь сводится к
одному переопределённому методу.

Предыстория. Хроника 21.08 сравнила два варианта на двенадцати эпохах:
фиксированный фильтр и предсказываемый по кадру. Числа оказались
неразличимы (0.4281 против 0.4255 при разбросе контроля 0.0144), но
поведение разное: предсказываемый СЕБЯ ВЫКЛЮЧИЛ — усиление сошлось к ровно
нулю на слое 16 и к 0.035 на слое 19; фиксированный к концу обучения ещё
рос (0.4241 → 0.4250 → 0.4281) и не нанёс раннего вреда (первая эпоха
0.3582 против 0.3584 у контроля). Тогда же было записано решение: на сто
эпох идёт фиксированный.

Запущен на сто эпох был предсказываемый по кадру — дважды, 54 часа. Оба
прогона провалили заранее записанный порог 31.08. То есть длинная проверка
досталась той версии, которая себя выключает, а решённая к проверке —
не проверялась ни разу. Этот скрипт закрывает пробел.

Отличия от версии 21.08: фильтр стоит в операторах ПОВЫШЕНИЯ РАЗРЕШЕНИЯ
(довод FreqFusion, TPAMI 2024), а не после слияния; считается depthwise-
свёрткой, а не через преобразование Фурье, поэтому исчезают и прежние
+17 % задержки, и островок полной точности внутри графа половинной.

Запуск:
    python training/train_lowpass.py --seed 42
    python training/train_lowpass.py --seed 42 --placement levels
    python training/train_lowpass.py --seed 42 --config training/configs/lowpass12.yaml

Выходы: runs/<run_name>/seed<N>/, копия весов в weights/<run_name>_seed<N>/.
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.neck_v3.fixed_lowpass import (  # noqa: E402
    has_lowpass, insert_lowpass, trainable_in_filters)
from training.train_baseline import (  # noqa: E402
    ensure_data_yaml, make_epoch_limiter,
)
from training.trainer_utils import (  # noqa: E402
    HeartbeatLogger, check_checkpoint_compatibility, setup_compute_flags,
)

NUM_CLASSES = 8


def make_lowpass_trainer(placement: str = 'upsample', kernel: int = 5,
                         sigma: float = 1.0, strength: float = 1.0):
    """Фабрика Trainer-класса с зафиксированными параметрами фильтра.

    Нужна потому, что model.train(trainer=...) принимает КЛАСС, а не
    экземпляр — параметризовать можно только через атрибуты класса.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class LowPassTrainer(DetectionTrainer):
        lp_placement = placement
        lp_kernel = kernel
        lp_sigma = sigma
        lp_strength = strength

        def get_model(self, cfg=None, weights=None, verbose=True):
            # resume: в модели фильтр уже стоит, второй раз вставлять нельзя
            if hasattr(weights, 'model') and has_lowpass(weights):
                return weights
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            idx = insert_lowpass(model, placement=self.lp_placement,
                                 kernel_size=self.lp_kernel, sigma=self.lp_sigma,
                                 strength=self.lp_strength)
            n_train = trainable_in_filters(model)
            print(f'[LP ] фильтр вставлен в слои {idx} '
                  f'(размещение: {self.lp_placement}, ядро {self.lp_kernel}, '
                  f'сигма {self.lp_sigma}, сила {self.lp_strength})')
            print(f'[LP ] обучаемых чисел в фильтрах: {n_train} '
                  f'{"— как и задумано" if n_train == 0 else "— ОШИБКА, должно быть 0"}')
            if n_train:
                raise RuntimeError(
                    f'фильтр обязан быть без обучаемых чисел, найдено {n_train}')
            return model

    return LowPassTrainer


def train_lowpass(config_path: str, seed: int, output_dir: str | None = None,
                  resume: bool = False, data_override: str | None = None,
                  placement: str | None = None, kernel: int | None = None,
                  sigma: float | None = None, strength: float | None = None,
                  stop_after: int | None = None) -> str:
    """Обучение одного сида YOLO + фиксированный фильтр. Возвращает путь к best.pt."""
    setup_compute_flags()
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    run_name = cfg.get('run_name') or 'lowpass'
    place = placement or cfg.get('lowpass_placement', 'upsample')
    k = kernel if kernel is not None else int(cfg.get('lowpass_kernel', 5))
    sg = sigma if sigma is not None else float(cfg.get('lowpass_sigma', 1.0))
    st = strength if strength is not None else float(cfg.get('lowpass_strength', 1.0))

    # Путь ОБЯЗАН быть абсолютным: относительный Ultralytics разрешает от
    # своего каталога, и результаты уезжают в runs/detect/runs/... — панель
    # их не видит, а супервизор убивает обучение как зависшее.
    run_root = Path(output_dir) if output_dir else PROJECT_ROOT / 'runs' / run_name
    if not run_root.is_absolute():
        run_root = (PROJECT_ROOT / run_root).resolve()
    run_dir = run_root / f'seed{seed}'
    weights_out = PROJECT_ROOT / 'weights' / f'{run_root.name}_seed{seed}'

    data_yaml = Path(data_override) if data_override else ensure_data_yaml(cfg)

    last = run_dir / 'weights' / 'last.pt'
    if resume and not last.exists():
        print(f'[WARN] --resume указан, но {last} нет — начинаю с нуля')
        resume = False
    if resume:
        start_weights, expect_nc = str(last), NUM_CLASSES
    else:
        start_weights, expect_nc = str(PROJECT_ROOT / cfg['weights']), 80
    if not check_checkpoint_compatibility(start_weights, expect_nc):
        raise SystemExit(1)

    model = YOLO(start_weights)
    hb = HeartbeatLogger(run_dir, block_epochs=cfg['block_epochs'])
    hb.attach(model)
    if stop_after:
        model.add_callback('on_fit_epoch_end', make_epoch_limiter(stop_after))

    model.train(
        trainer=make_lowpass_trainer(place, k, sg, st),
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
    print(f'[OK ] Сид {seed} ({run_name}): веса скопированы в {weights_out}')
    return str(weights_out / 'best.pt') if best_src.exists() else ''


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Обучение YOLO с фиксированным низкочастотным фильтром')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--config',
                    default=str(PROJECT_ROOT / 'training' / 'configs' / 'lowpass12.yaml'))
    ap.add_argument('--output', default=None, help='переопределить каталог запусков')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--data', default=None, help='свой data.yaml (smoke-прогоны)')
    ap.add_argument('--placement', default=None, choices=['upsample', 'levels'],
                    help='операторы повышения разрешения или выходы уровней')
    ap.add_argument('--kernel', type=int, default=None)
    ap.add_argument('--sigma', type=float, default=None)
    ap.add_argument('--strength', type=float, default=None,
                    help='сила вмешательства; 0 = точное тождество')
    ap.add_argument('--stop-after', type=int, default=None,
                    help='остановить после N эпох, расписание не менять')
    args = ap.parse_args()

    best = train_lowpass(args.config, args.seed, args.output, args.resume,
                         args.data, args.placement, args.kernel, args.sigma,
                         args.strength, args.stop_after)
    return 0 if best else 1


if __name__ == '__main__':
    sys.exit(main())
