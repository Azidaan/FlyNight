"""Сопоставимые обучаемые головы всех вариантов истории; ядроFlyVis не входит."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedGRU(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.cell = nn.GRUCell(inputs, hidden)

    def forward(self, sequence, valid):
        state = sequence.new_zeros((len(sequence), self.cell.hidden_size))
        for t in range(sequence.shape[1]):
            candidate = self.cell(sequence[:, t], state)
            state = torch.where(valid[:, t, None], candidate, state)
        return state


class HistoryPairHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 32), nn.ReLU())
        self.appearance = MaskedGRU(260, 32)
        self.observations = MaskedGRU(8, 16)
        self.geometry = nn.Sequential(nn.Linear(14, 32), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(112, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, batch):
        parts = [self.image(batch['image']), self.geometry(batch['geometry']),
            self.appearance(batch['appearance'], batch['appearance_valid']),
            self.observations(batch['observations'], batch['observation_valid'])]
        return self.fuse(torch.cat(parts, dim=1)).flatten()


def weighted_pair_loss(logits, targets, weights):
    """Неизвестные исключены; веса двух классов определяются толькоfit84."""
    known = targets >= 0
    if not known.any():
        return logits.sum()*0
    y = targets[known].long()
    loss = F.binary_cross_entropy_with_logits(logits[known], y.float(), reduction='none')
    return (loss*weights[y]).mean()


def corrected_cost(original, logits, eligible, alpha=.2):
    """Нулевая поправка побитно исходная. Новые рёбра не открываются."""
    if alpha == 0:
        return original.clone()
    delta = alpha*(.5-torch.sigmoid(logits))
    return torch.where(eligible, original+delta, original)
