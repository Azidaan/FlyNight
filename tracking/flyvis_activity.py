"""Кэш внутренней активности FlyVis и объектно-центрированное считывание.

Считывание определено в карточке ``flyvis_internal_readout_v1`` до опыта.
Модуль не зависит от рантайма FlyVis: он принимает уже полученные массивы
активности и координаты колонок и не обращается к разметке.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Типы клеток, которые читает декодер потока FlyVis (output_cell_types
# коннектома fib25-fib19_v2.2, проверено 16.09.2026 по рантайму).
OUTPUT_CELL_TYPES: tuple[str, ...] = (
    "T1", "T2", "T2a", "T3",
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9",
    "Tm16", "Tm20", "Tm28", "Tm30",
    "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18",
)
# Четыре кардинальных направления слоёв лобулярной пластинки.
DIRECTION_SUFFIXES: tuple[str, ...] = ("a", "b", "c", "d")
EPS = 1e-6


def type_column_indices(node_types: np.ndarray, cell_types: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    """Индексы узлов каждого типа в порядке коннектома (он совпадает с порядком рецепторов)."""
    node_types = np.asarray(node_types).astype(str)
    indices = {cell_type: np.flatnonzero(node_types == cell_type) for cell_type in cell_types}
    counts = {cell_type: index.size for cell_type, index in indices.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Типы имеют разное число колонок: {counts}")
    return indices


def mean_activity_by_type(activity: np.ndarray, indices: dict[str, np.ndarray]) -> np.ndarray:
    """(tiles, steps, nodes) -> (tiles, types, columns), среднее по внутренним шагам, float16."""
    activity = np.asarray(activity, dtype=np.float32)
    averaged = activity.mean(axis=1)
    stacked = np.stack([averaged[:, index] for index in indices.values()], axis=1)
    return stacked.astype(np.float16)


def save_activity_cache(path: Path, *, frames: np.ndarray, baseline: np.ndarray, points: np.ndarray, cell_types, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.asarray(frames, dtype=np.float16),
        baseline=np.asarray(baseline, dtype=np.float16),
        points=np.asarray(points, dtype=np.float32),
        cell_types=np.asarray(list(cell_types), dtype=str),
        **extra,
    )


def load_activity_cache(path: Path) -> dict:
    with np.load(path) as data:
        cache = {key: data[key] for key in data.files}
    cache["cell_types"] = [str(value) for value in cache["cell_types"]]
    return cache


def column_spacing(points: np.ndarray) -> float:
    """Медианное расстояние до ближайшей соседней колонки (шаг сетки в пикселях)."""
    sample = np.asarray(points, dtype=np.float32)
    if len(sample) < 2:
        return 0.0
    if len(sample) > 2000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 2000, replace=False)]
    distances = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def select_columns(points: np.ndarray, box: tuple[float, float, float, float], ring_scale: float = 1.5, min_margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Индексы колонок внутри рамки и в кольце вокруг неё.

    Кольцо — рамка, расширенная в ring_scale раз, но не меньше чем на
    min_margin (шаг сетки колонок) с каждой стороны; пустая рамка ->
    ближайшая колонка.
    """
    x1, y1, x2, y2 = (float(value) for value in box)
    px, py = points[:, 0], points[:, 1]
    inside_mask = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) / 2.0 + max((x2 - x1) / 2.0 * (ring_scale - 1.0), min_margin)
    half_h = (y2 - y1) / 2.0 + max((y2 - y1) / 2.0 * (ring_scale - 1.0), min_margin)
    ring_mask = (
        (px >= cx - half_w) & (px <= cx + half_w) & (py >= cy - half_h) & (py <= cy + half_h)
    ) & ~inside_mask
    inside = np.flatnonzero(inside_mask)
    if inside.size == 0:
        nearest = np.argmin(np.abs(px - cx) + np.abs(py - cy))
        inside = np.asarray([nearest])
        ring_mask[nearest] = False
    return inside, np.flatnonzero(ring_mask)


def _direction_energies(activity: np.ndarray, baseline: np.ndarray, columns: np.ndarray, type_index: dict[str, int]) -> dict:
    """Энергии четырёх направлений и ON/OFF по выбранным колонкам."""
    def rectified(cell_type: str) -> np.ndarray:
        row = type_index[cell_type]
        return np.maximum(activity[row, columns] - baseline[row, columns], 0.0)

    energies = {}
    on_total = off_total = 0.0
    for suffix in DIRECTION_SUFFIXES:
        on = float(rectified(f"T4{suffix}").mean())
        off = float(rectified(f"T5{suffix}").mean())
        energies[suffix] = on + off
        on_total += on
        off_total += off
    values = np.asarray([energies[s] for s in DIRECTION_SUFFIXES], dtype=np.float64)
    total = float(values.sum())
    probabilities = (values + EPS) / (total + len(values) * EPS)
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))
    ordered = np.sort(values)[::-1]
    return {
        "energy": energies,
        "on": on_total,
        "off": off_total,
        "total": total,
        "dominance": float((ordered[0] - ordered[1]) / (total + EPS)),
        "entropy": entropy,
        "net_x": float(energies["a"] - energies["b"]),
        "net_y": float(energies["c"] - energies["d"]),
    }


def box_readout_t4t5(activity: np.ndarray, baseline: np.ndarray, points: np.ndarray, box, *, cell_types=OUTPUT_CELL_TYPES, ring_scale: float = 1.5, min_margin: float = 0.0, expected_column_area: float | None = None) -> dict[str, float]:
    """Признаки T4/T5 по рамке: (types, columns) активность одного кадра и baseline той же формы."""
    activity = np.asarray(activity, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    type_index = {cell_type: row for row, cell_type in enumerate(cell_types)}
    inside, ring = select_columns(points, box, ring_scale, min_margin)
    features: dict[str, float] = {}
    for prefix, columns in (("inside", inside), ("ring", ring)):
        if columns.size == 0:
            stats = {"energy": {s: 0.0 for s in DIRECTION_SUFFIXES}, "on": 0.0, "off": 0.0, "total": 0.0, "dominance": 0.0, "entropy": 1.0, "net_x": 0.0, "net_y": 0.0}
        else:
            stats = _direction_energies(activity, baseline, columns, type_index)
        for suffix in DIRECTION_SUFFIXES:
            features[f"{prefix}_energy_{suffix}"] = stats["energy"][suffix]
        features[f"{prefix}_energy_on"] = stats["on"]
        features[f"{prefix}_energy_off"] = stats["off"]
        features[f"{prefix}_energy_total"] = stats["total"]
        features[f"{prefix}_dominance"] = stats["dominance"]
        features[f"{prefix}_entropy"] = stats["entropy"]
        features[f"{prefix}_net_x"] = stats["net_x"]
        features[f"{prefix}_net_y"] = stats["net_y"]
    features["contrast"] = features["inside_energy_total"] - features["ring_energy_total"]
    if expected_column_area is None:
        x1, y1, x2, y2 = box
        expected_column_area = max((x2 - x1) * (y2 - y1), EPS)
    features["coverage"] = float(inside.size) / float(expected_column_area) if expected_column_area else 0.0
    return features


def box_readout_all_types(activity: np.ndarray, baseline: np.ndarray, points: np.ndarray, box, *, cell_types=OUTPUT_CELL_TYPES, ring_scale: float = 1.5, min_margin: float = 0.0) -> dict[str, float]:
    """Средняя активность каждого типа (за вычетом baseline) внутри рамки и в кольце."""
    activity = np.asarray(activity, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    inside, ring = select_columns(points, box, ring_scale, min_margin)
    features: dict[str, float] = {}
    for prefix, columns in (("inside", inside), ("ring", ring)):
        block = activity[:, columns].mean(axis=1) if columns.size else np.zeros(len(cell_types), np.float32)
        for row, cell_type in enumerate(cell_types):
            features[f"{prefix}_{cell_type}"] = float(block[row])
    features["coverage"] = float(inside.size)
    return features
