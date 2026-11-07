"""Общий путь Б1: одна голова, причинная история, метки только в функции ошибки."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tracking.b1_real_inputs import corrected_identity_probability
from tracking.flyvis_temporal_residual import TemporalResidualAssociation, causal_pair_activities


class ReceptorHistory:
    """Контроль R1: 17 средних и 17 переходных каналов на той же истории."""
    def __init__(self, device):
        self.dev = torch.device(device)
        self.alpha = torch.as_tensor(1 / np.geomspace(1.5, 32., 17), device=self.dev, dtype=torch.float32)[None, :, None]

    def init(self, first):
        return first.reshape(first.shape[0], 1, -1).expand(-1, 17, -1).clone()

    def step(self, state, previous, current):
        value = current[:, None]
        state = state + (value - state) * self.alpha
        return state, torch.cat((state, value - state), 1)


def tensors_for_row(row, hexes, probability, device):
    """row содержит только наблюдаемые указатели; метки сюда не передаются."""
    n = len(row['choices']); columns = hexes.shape[-1]
    masks = torch.zeros((1, n, columns), device=device, dtype=torch.bool)
    for k, ids in enumerate(row['current_masks'], 1):
        masks[0, k, ids] = True
    reference = torch.zeros((1, columns), device=device, dtype=torch.bool)
    reference[0, row['reference_mask']] = True
    valid = torch.ones((1, n), device=device, dtype=torch.bool)
    real = valid.clone(); real[:, 0] = False
    base = torch.as_tensor(np.asarray(probability[:n])[None], device=device, dtype=torch.float64)
    return dict(base=base, masks=masks, reference_mask=reference, valid=valid, real=real,
                has_history=torch.tensor([row['has_history']], device=device))


def forward_residual(backbone, head, hexes, row, inputs, train_backbone):
    with torch.set_grad_enabled(torch.is_grad_enabled() and train_backbone):
        reference, current = causal_pair_activities(backbone, hexes, row['reference_index'])
    base = inputs['base']
    logits = torch.logit(torch.nan_to_num(base, nan=.5).clamp(1e-8, 1 - 1e-8)).float()
    _, delta = head(logits, reference, current, inputs['reference_mask'], inputs['masks'],
                    inputs['valid'], inputs['has_history'])
    return corrected_identity_probability(base, delta.double(), inputs['real']), delta


def identity_objective(updated, delta, labels, real, residual_weight=.01):
    """Среднее по известным реальным областям запроса; NULL и неизвестное исключены.

    Одна ошибка на запрос не даёт запросу с множеством соседей большего веса.
    Не подменяет полезность действия или HOTA. Регуляризация сдерживает поправку.
    """
    known = real & (labels >= 0)
    if not known.any():
        return None
    p = updated[known].clamp(1e-9, 1 - 1e-9)
    loss = nn.functional.binary_cross_entropy(p, labels[known].to(p.dtype))
    return loss + residual_weight * delta[known].square().mean()


def apply_gate_policy(gate_probability, geometry, threshold):
    """Чистое применение: исходное действие, явный NULL либо реальная область.

    Разметка и тип истинной ситуации отсутствуют во входах. Возврат None сохраняет
    исходную строку стоимости; 0 означает явный отказ, k>0 — область choices[k].
    """
    score = np.asarray(gate_probability, float)
    geometry = np.asarray(geometry)
    if score.ndim != 1 or geometry.ndim != 2 or len(score) != len(geometry):
        raise ValueError('Aligned action scores and observed geometry required')
    alternatives = geometry[:, -1] < .5
    scores = np.where(alternatives & np.isfinite(score), score, -np.inf)
    k = int(scores.argmax())
    return k if scores[k] >= threshold else None


def categorical_cost(original, choices, action):
    """Полная рамка меняет только строку назначения; None побитово сохраняет её."""
    source = np.asarray(original)
    if action is None:
        return source.copy()
    if choices[0] != -1 or not 0 <= action < len(choices):
        raise ValueError('Explicit NULL and unchanged candidate ordering required')
    row = np.ones_like(source)
    if action:
        row[int(choices[action])] = 0
    return row
