"""Метрика устойчивости Δ (Часть 2.6 плана).

Δ(метрика) = метрика(Normal) − метрика(Stress).
Чем меньше Δ, тем устойчивее метод к соответствующей деградации.
"""


def compute_stress_delta(metrics_normal: dict[str, float],
                         metrics_stress: dict[str, float]) -> dict[str, float]:
    """Δ по каждой метрике, присутствующей в обоих словарях (только числа)."""
    delta = {}
    for key, normal_val in metrics_normal.items():
        stress_val = metrics_stress.get(key)
        if isinstance(normal_val, (int, float)) and isinstance(stress_val, (int, float)):
            delta[key] = normal_val - stress_val
    return delta
