"""Регрессия «яркость видео → базовый порог трекера» (Этап 8, подэтап 8.3).

По результатам сеточного поиска для каждого видео известен оптимальный
track_high_thresh (лучший HOTA). Линейная регрессия по точкам
(brightness, optimal_thresh) даёт формулу базовой адаптации per-video:

    tau_base(видео) = a * brightness + b

Это «базовый уровень» адаптации, с которым в Этапе 9 сравнивается
LQA-guided (per-detection). Порог ограничивается диапазоном [TAU_MIN, TAU_MAX],
чтобы регрессия на краях распределения яркости не выдавала вырожденные значения.
"""
import numpy as np

TAU_MIN, TAU_MAX = 0.10, 0.70


def fit_brightness_to_threshold(
    brightness: list[float], optimal_thresh: list[float],
) -> tuple[float, float, float]:
    """Линейная регрессия. Возвращает (a, b, r2)."""
    x = np.asarray(brightness, dtype=np.float64)
    y = np.asarray(optimal_thresh, dtype=np.float64)
    if len(x) < 2:
        raise ValueError('Нужно >= 2 точки для регрессии')
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), r2


def apply_brightness_adaptive_threshold(
    brightness: float, a: float, b: float,
) -> float:
    """tau_base конкретного видео по формуле регрессии, с ограничением."""
    return float(np.clip(a * brightness + b, TAU_MIN, TAU_MAX))
