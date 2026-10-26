"""Обучаемая поправка к оценкам пар; разметка в прямой проход не передаётся.

FlyVis и базовый оценщик остаются внешними зависимостями. Активности должны
быть получены одним причинным проходом с сохранённым состоянием. Рамка
образца — наблюдаемая рамка трекера; маски кандидатов — реальные предложения.
Это прототип узла, а не готовое включение в XYWHByteTrack.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalResidualAssociation(nn.Module):
    """База + ограниченная поправка; слот 0 всегда означает отказ от назначения."""

    def __init__(self, channels: int = 34, hidden: int = 32):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, hidden), nn.Tanh())
        self.compare = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.Tanh())
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def pool(activity, masks):
        weights = masks.to(activity.dtype)
        return torch.einsum('bcn,bkn->bkc', activity, weights) / weights.sum(-1, keepdim=True).clamp_min(1)

    def forward(self, base_logits, reference, current, reference_mask, candidate_masks,
                candidate_valid, has_history):
        if base_logits.ndim != 2 or candidate_masks.shape[:2] != base_logits.shape:
            raise ValueError('Expected matching B,K base scores and candidate masks')
        if candidate_valid.shape != base_logits.shape or reference_mask.shape != reference.shape[::2]:
            raise ValueError('Invalid validity or reference mask shape')
        if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
            raise ValueError('Non-finite FlyVis activity')
        q = self.project(self.pool(reference, reference_mask[:, None]))
        k = self.project(self.pool(current, candidate_masks))
        q = q.expand_as(k)
        features = self.compare(torch.cat((q, k, q * k, (q - k).abs()), dim=-1))
        correction = torch.tanh(self.output(features).squeeze(-1))
        supported = (candidate_valid.bool() & candidate_masks.any(-1) & has_history[:, None].bool()
                     & reference_mask.any(-1)[:, None])
        supported = supported.clone()
        supported[:, 0] = False
        correction = torch.where(supported, correction, torch.zeros_like(correction))
        # No masking or renormalisation of the baseline: unavailable/historyless
        # slots retain exactly the score supplied by the original system.
        return base_logits + correction, correction


def causal_pair_activities(backbone, hexes, reference_index: int):
    """Run through the last supplied observation, retaining gradients to FlyVis.

    Caller must truncate hexes at the current observation and keep the geometric
    interpretation of each mask in its own observation's coordinates. This helper
    never selects a reference by ground truth or reads later frames.
    """
    if hexes.ndim != 3 or not 0 < reference_index < hexes.shape[1] - 1:
        raise ValueError('Reference must precede current observation and follow fade-in')
    state = backbone.init(hexes[:, :1])
    reference = None
    for index in range(1, hexes.shape[1]):
        state, activity = backbone.step(state, hexes[:, index - 1], hexes[:, index])
        if index == reference_index:
            reference = activity
    return reference, activity
