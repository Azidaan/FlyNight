"""Обвязка тренера (Этап 3): heartbeat-лог, блочная оценка, совместимость чекпойнтов.

Всё подключается к Ultralytics через штатные callbacks — сам тренер не патчится.
"""
import json
import threading
import time
from pathlib import Path


def setup_compute_flags() -> None:
    """Глобальные настройки вычислений (Часть 2.7 плана): TF32 + cudnn.benchmark.

    AMP Ultralytics включает сам (amp=True по умолчанию).
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def check_checkpoint_compatibility(checkpoint_path: str, expected_nc: int) -> bool:
    """Проверяет чекпойнт до загрузки в тренер: число классов должно совпадать.

    Возвращает True/False и печатает понятное сообщение вместо глухого
    RuntimeError из недр PyTorch при несовпадении головы.
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ckpt.get('model') or ckpt.get('ema')
    nc = None
    if model is not None:
        nc = getattr(model, 'nc', None) or (getattr(model, 'yaml', {}) or {}).get('nc')
    if nc is None:
        print(f'[WARN] {checkpoint_path}: число классов не определено — '
              f'продолжаем на свой риск')
        return True
    if int(nc) != expected_nc:
        print(f'[FAIL] Чекпойнт {checkpoint_path} обучен на {nc} классов, '
              f'конфигурация ожидает {expected_nc}.\n'
              f'       Либо это чекпойнт другого эксперимента, либо неверный '
              f'yaml данных. Обучение не запущено.')
        return False
    print(f'[OK ] Чекпойнт совместим: {nc} классов')
    return True


class HeartbeatLogger:
    """Живое логирование: не даёт спутать долгий прогрев с зависанием.

    Что делает (по плану Этапа 3):
    - отдельная нить печатает heartbeat каждые interval_sec, если тренер молчит;
    - логирует ожидание первого батча (CUDA-прогрев + cudnn.benchmark);
    - после каждой эпохи — сводка: loss-фаза, mAP, лучший mAP, ETA;
    - каждые block_epochs эпох сохраняет метрики в eval_metrics_epoch_N.json.
    """

    def __init__(self, output_dir: str | Path,
                 interval_sec: int = 30, block_epochs: int = 10) -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.interval = interval_sec
        self.block = block_epochs
        self.phase = 'инициализация'
        self.last_msg = time.monotonic()
        self.t_start = time.monotonic()
        self.t_epoch_start: float | None = None
        self.epoch_durations: list[float] = []
        self.first_batch_seen = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)

    # ——— служебное ———
    def _say(self, msg: str) -> None:
        elapsed = time.monotonic() - self.t_start
        print(f'[HB {elapsed/60:6.1f}м] {msg}', flush=True)
        self.last_msg = time.monotonic()

    def _beat(self) -> None:
        while not self._stop.wait(5):
            if time.monotonic() - self.last_msg >= self.interval:
                self._say(f'работаю: {self.phase}')

    # ——— callbacks Ultralytics ———
    def attach(self, model) -> None:
        model.add_callback('on_pretrain_routine_start', self._on_setup)
        model.add_callback('on_train_start', self._on_train_start)
        model.add_callback('on_train_batch_start', self._on_batch_start)
        model.add_callback('on_train_batch_end', self._on_batch_end)
        model.add_callback('on_train_epoch_start', self._on_epoch_start)
        model.add_callback('on_fit_epoch_end', self._on_fit_epoch_end)
        model.add_callback('on_train_end', self._on_train_end)
        self._thread.start()

    def _on_setup(self, trainer) -> None:
        self.phase = 'подготовка данных и модели'
        self._say('старт: подготовка тренера (сборка датасета может занять минуты)')

    def _on_train_start(self, trainer) -> None:
        self.phase = 'ожидание первого батча'
        self._say('обучение запущено; первый батч может готовиться десятки секунд '
                  '(CUDA-прогрев, cudnn.benchmark) — это не зависание')

    def _on_batch_start(self, trainer) -> None:
        self.phase = f'эпоха {trainer.epoch + 1}: forward/backward'

    def _on_batch_end(self, trainer) -> None:
        if not self.first_batch_seen:
            self.first_batch_seen = True
            self._say('первый батч пройден — конвейер работает')

    def _on_epoch_start(self, trainer) -> None:
        self.t_epoch_start = time.monotonic()

    def _on_fit_epoch_end(self, trainer) -> None:
        epoch = trainer.epoch + 1
        if epoch > trainer.epochs:   # финальная валидация best.pt после обучения
            self._say('финальная валидация best.pt')
            return
        if self.t_epoch_start is not None:
            self.epoch_durations.append(time.monotonic() - self.t_epoch_start)
        metrics = {k: float(v) for k, v in (trainer.metrics or {}).items()}
        map50 = metrics.get('metrics/mAP50(B)')
        best = float(trainer.best_fitness or 0)
        eta_min = 0.0
        if self.epoch_durations:
            mean = sum(self.epoch_durations[-5:]) / len(self.epoch_durations[-5:])
            eta_min = mean * (trainer.epochs - epoch) / 60
        self._say(f'эпоха {epoch}/{trainer.epochs}: '
                  f'mAP50={map50:.4f} · best_fitness={best:.4f} · '
                  f'ETA ~{eta_min/60:.1f} ч' if map50 is not None else
                  f'эпоха {epoch}/{trainer.epochs} завершена · ETA ~{eta_min/60:.1f} ч')
        if epoch % self.block == 0 or epoch == trainer.epochs:
            out = self.out / f'eval_metrics_epoch_{epoch}.json'
            out.write_text(json.dumps({'epoch': epoch, 'metrics': metrics,
                                       'best_fitness': best},
                                      ensure_ascii=False, indent=2),
                           encoding='utf-8')
            self._say(f'блок {epoch // self.block}: метрики сохранены в {out.name}')

    def _on_train_end(self, trainer) -> None:
        self.phase = 'завершение'
        self._say(f'обучение завершено; best={trainer.best}')
        self._stop.set()
