"""Наблюдаемые признаки сохранённой базы Б1, пригодные для работы в трекере."""
from __future__ import annotations
import numpy as np


def relative_features(features, mask):
    """Тот же порядок diff/rank, что у исходного преобразования v3."""
    real = mask.copy(); real[0] = False
    out = []
    for c in range(features.shape[-1]):
        values = np.where(real, np.nan_to_num(features[:, c], nan=-1e9), -np.inf)
        best = np.empty_like(values)
        for k in range(len(values)):
            other = values.copy(); other[k] = -np.inf; best[k] = other.max()
        diff = np.where(np.isfinite(best) & real & np.isfinite(features[:, c]), features[:, c] - best, np.nan)
        rank = np.where(real, (-values).argsort().argsort(), np.nan)
        out += [diff, rank]
    return np.stack(out, -1).astype(np.float32)


def appearance_features(geo, mask, region, coarse_exemplar, snapped_exemplar, mean, projection, previous=None, gap=0):
    """Нет меток: геометрия, два наблюдаемых образца и предыдущий наблюдаемый выбор.

    Возвращает 98 признаков и банк прошлого для сохранения после этого запроса.
    region содержит NULL в слоте0 и те же дополненные слоты, что исходная база.
    """
    geo = np.asarray(geo, np.float32); mask = np.asarray(mask, bool)
    region = np.asarray(region, np.float32); width = len(mask)
    valid = mask.copy(); valid[0] = False
    tagged_geometry = np.full((width, 10), np.nan, np.float32); tagged_geometry[:, 9] = 0
    past = np.full(256, np.nan, np.float32); bank = past.copy()
    if previous is not None:
        if previous.get('geometry') is not None:
            tagged_geometry[mask] = np.concatenate([
                geo[mask, :8] - previous['geometry'],
                np.full((int(mask.sum()), 1), gap, np.float32),
                np.ones((int(mask.sum()), 1), np.float32)], 1)
        if previous.get('embedding') is not None: past = previous['embedding']
        bank = previous['bank'].copy()
        if np.isfinite(past).all(): bank = past.copy() if not np.isfinite(bank).all() else .5 * past + .5 * bank
    similarities = np.full((width, 3), np.nan, np.float32)
    for column, reference in enumerate((coarse_exemplar, past, bank)):
        norm = reference / max(float(np.linalg.norm(reference)), 1e-9)
        similarities[:, column] = np.where(valid & np.isfinite(norm).all(), (region * norm).sum(-1), np.nan)
    cosine = np.where(valid, (region * snapped_exemplar).sum(-1), np.nan)[:, None]
    projected_region = (region - mean) @ projection
    projected_exemplar = (snapped_exemplar - mean) @ projection
    products = np.where(valid[:, None], projected_region * projected_exemplar, np.nan)
    initial = np.concatenate([geo[:, :8], tagged_geometry, similarities, cosine], -1)
    return np.concatenate([initial, relative_features(initial, mask), products], -1), bank


def observe_tag(geo, mask, region, bank):
    """Сохраняется исходное назначение трекера до применения поправки."""
    candidates = np.flatnonzero(mask & (geo[:, -1] > .5))
    k = int(candidates[0]) if len(candidates) else None
    return dict(geometry=geo[k, :8].copy() if k is not None else None,
                embedding=region[k].copy() if k is not None and k > 0 else None, bank=bank.copy())


def gate_features(geo, mask, probability):
    """Исходные признаки дерева действий; ширина дополнения сохранена."""
    p = np.nan_to_num(probability, nan=-1.)
    actual = np.where(mask, p, -1.); others = np.empty_like(p)
    for k in range(len(p)):
        one = actual.copy(); one[k] = -1.; others[k] = one.max()
    selected = np.where(geo[:, -1] > .5, p, -np.inf).max()
    selected = selected if np.isfinite(selected) else -1.
    rank = (-actual).argsort().argsort().astype(np.float32)
    return np.concatenate([geo, probability[:, None], others[:, None],
                           np.full((len(p), 1), selected), rank[:, None], (p - others)[:, None]], -1)
