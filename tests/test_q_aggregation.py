"""Unit-тесты проекции координат и агрегации q_i (Этап 9, tracking/q_aggregation).

Проекция letterbox — самое хрупкое место проброса Q: ошибка в gain/паддинге
смещает область агрегации, и q_i молча считается по чужим ячейкам.
Проверяются: известные значения для BDD-геометрии (1280x720 -> 384x640),
крайние случаи (рамка у границы, субпиксельная рамка), сами агрегации
и broadcast-семантика compute_q_per_detection.

Запуск: python tests/test_q_aggregation.py  (CPU, без модели)
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.q_aggregation import (  # noqa: E402
    aggregate_q, box_to_qmap_coords, compute_q_per_detection,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def test_projection() -> None:
    # BDD-геометрия: 1280x720 -> letterbox 384x640, gain 0.5, pad_y 12.
    orig, inp = (720, 1280), (384, 640)

    cells = box_to_qmap_coords(np.array([0., 0., 1280., 720.]), orig, inp, 8)
    check('full-frame на P3', cells == (0, 1, 80, 47), str(cells))

    cells = box_to_qmap_coords(np.array([0., 0., 1280., 720.]), orig, inp, 32)
    check('full-frame на P5', cells == (0, 0, 20, 12), str(cells))

    # Центральная рамка 128x128 px оригинала: центр (640,360)->вход (320,192),
    # на P3: x [36..44], y [20..28].
    cells = box_to_qmap_coords(np.array([576., 296., 704., 424.]), orig, inp, 8)
    check('центральная рамка на P3', cells == (36, 20, 44, 28), str(cells))

    # Субпиксельная рамка (2x2 px) -> минимум 1x1 ячейка, без выхода за карту.
    cells = box_to_qmap_coords(np.array([1278., 718., 1280., 720.]), orig, inp, 32)
    x1, y1, x2, y2 = cells
    check('субпиксельная рамка в углу: 1x1 внутри карты',
          x2 - x1 >= 1 and y2 - y1 >= 1 and x2 <= 20 and y2 <= 12, str(cells))

    # Квадратный вход без паддинга по обеим осям (тождественный масштаб).
    cells = box_to_qmap_coords(np.array([0., 0., 640., 640.]),
                               (640, 640), (640, 640), 8)
    check('квадрат без letterbox', cells == (0, 0, 80, 80), str(cells))


def test_aggregations() -> None:
    q = torch.zeros(1, 1, 10, 10)
    q[0, 0, 2:4, 2:4] = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    cells = (2, 2, 4, 4)
    check('mean', abs(aggregate_q(q, cells, 'mean') - 0.5) < 1e-6)
    check('min', abs(aggregate_q(q, cells, 'min') - 0.2) < 1e-6)
    check('median', abs(aggregate_q(q, cells, 'median') - 0.4) < 1e-6,
          f'{aggregate_q(q, cells, "median"):.3f} (torch: нижняя медиана)')
    check('пустая область -> нейтральное 1.0',
          aggregate_q(q, (5, 5, 5, 5), 'mean') == 1.0)


def test_compute_per_detection() -> None:
    # Две карты: P3 = 0.25 всюду, P5 = 0.75 всюду; проверяем маршрутизацию
    # по страйду и нейтральное q=1 для уровня без карты.
    q_maps = {8: torch.full((1, 1, 48, 80), 0.25),
              32: torch.full((1, 1, 12, 20), 0.75)}
    boxes = np.array([[100., 100., 300., 300.]] * 3, np.float32)
    strides = np.array([8, 32, 16], np.int16)
    q = compute_q_per_detection(q_maps, boxes, strides, (720, 1280),
                                (384, 640), 'mean')
    check('маршрутизация по страйду',
          abs(q[0] - 0.25) < 1e-6 and abs(q[1] - 0.75) < 1e-6,
          f'{q.tolist()}')
    check('уровень без карты -> нейтральное 1.0', q[2] == 1.0)


def main() -> int:
    test_projection()
    test_aggregations()
    test_compute_per_detection()
    print(f'\n--- {"OK: все проверки q_aggregation пройдены" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
