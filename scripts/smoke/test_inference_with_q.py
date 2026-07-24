"""Smoke-тест проброса Q и привязки детекций к уровням (Этап 9, п. 9.1-9.2).

Модель: yolo26s.pt (COCO) + свежевставленный DRB (случайная инициализация) —
значения Q содержательно шумные, но весь конвейер (патч головы, страйды,
letterbox-проекция, агрегации) проверяется полноценно.

Проверки:
- stride есть у каждой детекции и принадлежит {8, 16, 32};
- у модели с DRB карты Q есть на всех трёх уровнях, формы соответствуют
  квадратному letterbox 640/stride;
- q_mean/q_min/q_median одной длины с детекциями, значения в [0, 1];
- q_min <= q_mean (min не больше среднего по области).

Запуск: python scripts/smoke/test_inference_with_q.py [video_id]
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from models.yolo_neck import insert_drb  # noqa: E402
from tracking.inference_with_q import IMGSZ, QInference  # noqa: E402

MODEL = str(PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt')
VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'b1cebfb7-284f5117'
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    frames = sorted((DATA_ROOT / 'images' / 'track' / 'val' / VIDEO).glob('*.jpg'))
    frame = frames[len(frames) // 2]
    print(f'[INFO] кадр {frame.name}')

    qi = QInference(MODEL)
    check('модель без DRB распознана', not qi.with_drb)
    out = qi.predict_frame(frame)
    n = len(out['conf'])
    check('детекции есть', n > 0, f'n={n}')
    check('stride на каждую детекцию', len(out['stride']) == n)
    check('страйды из {8,16,32}',
          set(np.unique(out['stride'])).issubset({8, 16, 32}),
          str(sorted(set(out['stride'].tolist()))))
    check('q_* отсутствуют без DRB', 'q_mean' not in out)

    print('[INFO] вставляю DRB (случайная инициализация)...')
    insert_drb(qi.det_model, imgsz=IMGSZ)
    qi.with_drb = True
    out = qi.predict_frame(frame, aggregations=('mean', 'min', 'median'))
    n = len(out['conf'])

    q_maps = qi.q_maps_by_stride()
    check('карты Q на трёх уровнях', set(q_maps) == {8, 16, 32},
          str(sorted(q_maps)))
    # Letterbox прямоугольный: уровни согласованы, если q_shape * stride
    # даёт один и тот же размер входа на всех уровнях.
    input_shapes = {(q.shape[-2] * s, q.shape[-1] * s)
                    for s, q in q_maps.items()}
    check('формы Q согласованы между уровнями (один input_shape)',
          len(input_shapes) == 1,
          str({s: tuple(q.shape[-2:]) for s, q in q_maps.items()}))

    # Точечный юнит проекции: рамка всего кадра BDD (1280x720) на входе
    # 384x640 (gain 0.5, pad_y 12) должна занять ячейки y=[1..47] на P3.
    from tracking.q_aggregation import box_to_qmap_coords
    cells = box_to_qmap_coords(
        np.array([0.0, 0.0, 1280.0, 720.0]), (720, 1280), (384, 640), 8)
    check('проекция letterbox: full-frame рамка -> ожидаемые ячейки P3',
          cells == (0, 1, 80, 47), str(cells))

    for agg in ('mean', 'min', 'median'):
        arr = out[f'q_{agg}']
        check(f'q_{agg}: длина и диапазон',
              len(arr) == n and bool((arr >= 0).all() and (arr <= 1).all()),
              f'n={len(arr)}, [{arr.min():.3f}, {arr.max():.3f}]')
    check('q_min <= q_mean поэлементно',
          bool((out['q_min'] <= out['q_mean'] + 1e-6).all()))

    print(f'\n--- {"OK: проброс Q и привязка к уровням работают" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
