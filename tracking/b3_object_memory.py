"""Б3: признаки внешнего вида подаются в новую память через обучаемый вход."""
from __future__ import annotations
import torch
from torch import nn
from tracking.b1_real_inputs import corrected_identity_probability


class ObjectMemoryBranch(nn.Module):
    def __init__(self, columns=721):
        super().__init__()
        # 256 appearance components + observable presence flag. No identity label.
        self.input = nn.Linear(257, columns)
        self.region = nn.Linear(256, 34)
        self.normalize = nn.LayerNorm(34)
        self.compare = nn.Sequential(nn.Linear(136, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.compare[-1].weight); nn.init.zeros_(self.compare[-1].bias)

    def forward(self, observed_history, candidates, base, backbone):
        if observed_history.ndim != 3 or observed_history.shape[1] < 2 or observed_history.shape[-1] != 257:
            raise ValueError('At least two observable memory events required')
        receptor = torch.sigmoid(self.input(observed_history))
        state = backbone.init(receptor[:, :1])
        activity = None
        for i in range(1, receptor.shape[1]):state, activity = backbone.step(state, receptor[:, i-1], receptor[:, i])
        memory = self.normalize(activity.mean(-1))[:, None]
        current = self.normalize(self.region(candidates))
        memory = memory.expand_as(current)
        delta = torch.tanh(self.compare(torch.cat([memory, current, memory * current, (memory-current).abs()], -1)).squeeze(-1))
        return corrected_identity_probability(base, delta.double(), torch.ones_like(base,dtype=torch.bool)),delta
