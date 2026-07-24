"""Smoke-тест трёх версий YOLO и выбор рабочей версии (Этап 1, блок 3).

Для каждой из YOLO26 / YOLOv11 / YOLOv8 (масштаб s) последовательно:
1. Загрузка предобученной модели (веса кладутся в weights/pretrained/).
2. Инференс на одном случайном изображении 640x640.
3. Обучение на 10 батчах фиктивного мини-датасета (40 изображений с
   нарисованными прямоугольниками + YOLO txt-разметка; сырые тензоры
   model.train() не принимает). Проверка: loss уменьшился, NaN нет.
4. Замер FPS на батче 16: прогрев 5 проходов, замер 20 проходов
   через torch.cuda.synchronize. Цель для YOLO26 — не менее 50 FPS.

Падение одного шага не останавливает скрипт: ошибка пишется в поле error,
последующие шаги версии пропускаются, скрипт переходит к следующей версии.

Выходы:
- experiments/results/smoke_test_yolo.csv — таблица результатов;
- notes/yolo_version_choice.md — решение (первая версия из списка
  приоритета [YOLO26, YOLOv11, YOLOv8], прошедшая все 4 шага).
"""
import csv
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = PROJECT_ROOT / 'weights' / 'pretrained'
RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'smoke_test_yolo.csv'
CHOICE_MD = PROJECT_ROOT / 'notes' / 'yolo_version_choice.md'
DUMMY_DIR = PROJECT_ROOT / 'data' / 'dummy_smoke'
RUNS_DIR = PROJECT_ROOT / 'runs' / 'smoke'

# Приоритет версий — Часть 2.8 плана: YOLO26 -> YOLOv11 -> YOLOv8.
VERSIONS = [
    ('YOLO26', 'yolo26s.pt'),
    ('YOLOv11', 'yolo11s.pt'),
    ('YOLOv8', 'yolov8s.pt'),
]
FPS_TARGET = 50.0
BATCH = 16
N_WARMUP, N_TIMED = 5, 20


def setup_compute_flags() -> None:
    """Глобальные настройки вычислений из Части 2.7 плана."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def ensure_weights(weights_name: str) -> Path:
    """Возвращает путь к весам в weights/pretrained/, скачивая при отсутствии.

    Ultralytics качает веса в текущую директорию, поэтому на время скачки
    переходим в WEIGHTS_DIR — все веса остаются внутри проекта.
    """
    target = WEIGHTS_DIR / weights_name
    if target.exists():
        return target
    from ultralytics import YOLO
    cwd = os.getcwd()
    try:
        os.chdir(WEIGHTS_DIR)
        YOLO(weights_name)  # триггерит скачивание в текущую директорию
    finally:
        os.chdir(cwd)
    if not target.exists():
        raise FileNotFoundError(f'Веса {weights_name} не скачались в {WEIGHTS_DIR}')
    return target


def make_dummy_dataset() -> Path:
    """Фиктивный мини-датасет: шумовые изображения с прямоугольниками + txt-разметка."""
    import cv2
    rng = np.random.default_rng(42)
    yaml_path = DUMMY_DIR / 'dummy.yaml'
    if yaml_path.exists():
        return yaml_path

    for split, n in (('train', 40), ('val', 8)):
        img_dir = DUMMY_DIR / 'images' / split
        lbl_dir = DUMMY_DIR / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
            lines = []
            for _ in range(int(rng.integers(1, 4))):
                w, h = int(rng.integers(60, 200)), int(rng.integers(60, 200))
                x1 = int(rng.integers(0, 640 - w))
                y1 = int(rng.integers(0, 640 - h))
                color = tuple(int(c) for c in rng.integers(0, 255, 3))
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, -1)
                cx, cy = (x1 + w / 2) / 640, (y1 + h / 2) / 640
                lines.append(f'0 {cx:.6f} {cy:.6f} {w / 640:.6f} {h / 640:.6f}')
            cv2.imwrite(str(img_dir / f'{i:03d}.jpg'), img)
            (lbl_dir / f'{i:03d}.txt').write_text('\n'.join(lines), encoding='utf-8')

    yaml_path.write_text(
        f'path: {DUMMY_DIR.as_posix()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'names:\n  0: obj\n',
        encoding='utf-8',
    )
    return yaml_path


def smoke_test_one_version(version_name: str, weights_path: str) -> dict:
    """Прогоняет 4 проверки для одной версии. Исключения ловит в поле error."""
    import torch
    from ultralytics import YOLO

    result = {
        'version': version_name, 'weights': Path(weights_path).name,
        'load_ok': False, 'infer_ok': False, 'train_ok': False,
        'loss_first': '', 'loss_last': '', 'fps': '', 'fps_ok': False,
        'all_passed': False, 'error': '',
    }
    step = 'load'
    try:
        model = YOLO(str(weights_path))
        result['load_ok'] = True

        step = 'inference'
        img = np.random.default_rng(0).integers(0, 255, (640, 640, 3), dtype=np.uint8)
        model.predict(img, imgsz=640, device=0, verbose=False)
        result['infer_ok'] = True

        step = 'train'
        yaml_path = make_dummy_dataset()
        losses: list[float] = []

        def on_train_batch_end(trainer):
            losses.append(float(trainer.loss.detach().cpu()))

        train_model = YOLO(str(weights_path))
        train_model.add_callback('on_train_batch_end', on_train_batch_end)
        train_model.train(
            data=str(yaml_path), epochs=1, imgsz=640, batch=4, workers=0,
            device=0, verbose=False, plots=False, val=False, save=False,
            seed=42, project=str(RUNS_DIR), name=f'{version_name}_train',
            exist_ok=True,
        )
        if not losses:
            raise RuntimeError('callback не собрал значения loss')
        nan_found = any(not np.isfinite(v) for v in losses)
        decreased = losses[-1] < losses[0]
        result['loss_first'] = f'{losses[0]:.4f}'
        result['loss_last'] = f'{losses[-1]:.4f}'
        if nan_found:
            raise RuntimeError(f'NaN/Inf в loss: {losses}')
        if not decreased:
            raise RuntimeError(
                f'loss не уменьшился за {len(losses)} батчей: '
                f'{losses[0]:.4f} -> {losses[-1]:.4f}'
            )
        result['train_ok'] = True
        del train_model
        torch.cuda.empty_cache()

        step = 'fps'
        module = YOLO(str(weights_path)).model.cuda().eval().float()
        x = torch.rand(BATCH, 3, 640, 640, device='cuda')
        with torch.inference_mode():
            for _ in range(N_WARMUP):
                module(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                module(x)
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        fps = BATCH * N_TIMED / dt
        result['fps'] = f'{fps:.1f}'
        result['fps_ok'] = fps >= FPS_TARGET
        if not result['fps_ok']:
            raise RuntimeError(f'FPS {fps:.1f} ниже целевых {FPS_TARGET}')

        result['all_passed'] = True
        del module
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - смысл скрипта: собрать все сбои
        result['error'] = f'[{step}] {type(exc).__name__}: {exc}'
        traceback.print_exc()
    return result


def write_results_to_csv(results: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def choose_working_version(results: list[dict]) -> str | None:
    for r in results:
        if r['all_passed']:
            return r['version']
    return None


def write_choice_md(results: list[dict], chosen: str | None) -> None:
    lines = [
        '# Выбор версии YOLO',
        '',
        f'Дата: {date.today().isoformat()}. Железо: RTX 5070, окружение из notes/environment.md.',
        '',
        'Smoke-тест по Этапу 1 плана: загрузка, инференс, 10 батчей обучения',
        f'(loss убывает, без NaN), FPS на батче {BATCH} (цель ≥ {FPS_TARGET:.0f}).',
        '',
        '| Версия | Загрузка | Инференс | Обучение | loss 1→10 | FPS | Итог |',
        '|--------|----------|----------|----------|-----------|-----|------|',
    ]
    for r in results:
        mark = lambda ok: '✅' if ok else '❌'  # noqa: E731
        lines.append(
            f"| {r['version']} | {mark(r['load_ok'])} | {mark(r['infer_ok'])} "
            f"| {mark(r['train_ok'])} | {r['loss_first']}→{r['loss_last']} "
            f"| {r['fps']} | {'ПРОШЛА' if r['all_passed'] else r['error'][:80]} |"
        )
    lines += ['']
    if chosen:
        lines += [
            f'## Решение: **{chosen}**',
            '',
            f'{chosen} — первая версия в порядке приоритета (YOLO26 → YOLOv11 → YOLOv8,'
            ' Часть 2.8 плана), прошедшая все четыре проверки. Все последующие этапы'
            f' используют {chosen}; возврат к более новой версии в середине работы'
            ' не предусмотрен.',
        ]
    else:
        lines += [
            '## Решение: НИ ОДНА ВЕРСИЯ НЕ ПРОШЛА',
            '',
            'Требуется ручной разбор (см. поле error в CSV). План Б: torchvision-реализация.',
        ]
    CHOICE_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    setup_compute_flags()
    results = []
    for version_name, weights_name in VERSIONS:
        print(f'\n=== {version_name} ({weights_name}) ===')
        try:
            weights_path = ensure_weights(weights_name)
        except Exception as exc:  # noqa: BLE001
            results.append({
                'version': version_name, 'weights': weights_name,
                'load_ok': False, 'infer_ok': False, 'train_ok': False,
                'loss_first': '', 'loss_last': '', 'fps': '', 'fps_ok': False,
                'all_passed': False, 'error': f'[download] {type(exc).__name__}: {exc}',
            })
            continue
        r = smoke_test_one_version(version_name, str(weights_path))
        print({k: v for k, v in r.items() if v != ''})
        results.append(r)

    write_results_to_csv(results, str(RESULTS_CSV))
    chosen = choose_working_version(results)
    write_choice_md(results, chosen)

    print(f'\nCSV: {RESULTS_CSV}')
    print(f'Решение: {CHOICE_MD}')
    print(f'Выбранная версия: {chosen or "НЕТ — все версии провалились"}')
    return 0 if chosen else 1


if __name__ == '__main__':
    sys.exit(main())
