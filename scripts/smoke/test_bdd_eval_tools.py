"""Smoke-тест инструментов оценки трекинга (Этап 1, блок 4).

План А: официальные bdd100k-mot eval tools (пакет bdd100k + scalabel) —
нативный формат BDD100K, конвертация не нужна.
План Б: TrackEval (клон в third_party/TrackEval) — стандартная реализация
HOTA/IDF1/MOTA, формат MOTChallenge.

Проверки:
1. bdd100k: импорт модуля оценки и запуск CLI --help (если установлен).
2. TrackEval: синтетический пример — две идеальные траектории на 10 кадрах,
   ожидаемый HOTA == 1.0 (проверка, что метрика считается и математика верна).

Скрипт успешен, если работает хотя бы один инструмент (какой — печатается,
результат фиксируется в notes/environment.md).
"""
import subprocess
import sys

import numpy as np


def check_bdd100k() -> bool:
    try:
        import bdd100k  # noqa: F401
    except ImportError as exc:
        print(f'[INFO] пакет bdd100k не установлен: {exc}')
        return False
    proc = subprocess.run(
        [sys.executable, '-m', 'bdd100k.eval.run', '--help'],
        capture_output=True, text=True, timeout=120,
    )
    ok = proc.returncode == 0
    print(f'[{"OK " if ok else "FAIL"}] bdd100k.eval.run --help -> код {proc.returncode}')
    if not ok:
        print(proc.stderr[-2000:])
    return ok


def check_trackeval_synthetic() -> bool:
    try:
        from trackeval.metrics import HOTA
    except ImportError as exc:
        print(f'[INFO] пакет trackeval не установлен: {exc}')
        return False

    # Синтетика: 2 объекта, 10 кадров, трекер идеально совпадает с GT.
    n_frames, n_ids = 10, 2
    data = {
        'num_tracker_dets': n_frames * n_ids,
        'num_gt_dets': n_frames * n_ids,
        'num_gt_ids': n_ids,
        'num_tracker_ids': n_ids,
        'gt_ids': [np.arange(n_ids) for _ in range(n_frames)],
        'tracker_ids': [np.arange(n_ids) for _ in range(n_frames)],
        'similarity_scores': [np.eye(n_ids) for _ in range(n_frames)],
    }
    res = HOTA().eval_sequence(data)
    hota = float(np.mean(res['HOTA']))
    ok = abs(hota - 1.0) < 1e-6
    print(f'[{"OK " if ok else "FAIL"}] TrackEval, синтетический идеальный трекинг: HOTA={hota:.4f} (ожидание 1.0)')
    return ok


def main() -> int:
    bdd_ok = check_bdd100k()
    te_ok = check_trackeval_synthetic()

    if bdd_ok:
        print('\n--- OK: официальные bdd100k eval tools работают (План А) ---')
        return 0
    if te_ok:
        print('\n--- OK: TrackEval работает (План Б; bdd100k tools недоступны) ---')
        return 0
    print('\n--- ПРОВЕРКА НЕ ПРОЙДЕНА: ни один инструмент оценки не работает ---')
    return 1


if __name__ == '__main__':
    sys.exit(main())
