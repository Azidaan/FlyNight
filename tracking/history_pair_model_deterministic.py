"""Тот же8×8→4×4неперекрывающийсясреднийпул с определённым CUDAградиентом."""
from torch import nn
from tracking.history_pair_model import HistoryPairHead


class DeterministicHistoryPairHead(HistoryPairHead):
    def __init__(self):
        super().__init__()
        self.image[6] = nn.AvgPool2d(2, stride=2)
