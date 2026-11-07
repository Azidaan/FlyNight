"""Б2: причинная ветвь между признаками изображения и решением детектора."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TemporalDetectionBranch(nn.Module):
    """Общий вход и выход для FlyVis и контролей; ядро задаётся отдельно."""
    def __init__(self, channels, points, eye_size):
        super().__init__()
        points = np.asarray(points, np.float32)
        self.register_buffer('sample_grid', torch.as_tensor(points / eye_size * 2 - 1)[None, None])
        self.input = nn.Conv2d(channels, 1, 1)
        self.output = nn.Conv2d(34, channels, 1)
        nn.init.constant_(self.input.weight, 1 / channels); nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.points = points / eye_size
        self.raster_indices = {}

    def receptors(self, features):
        image = torch.sigmoid(self.input(features))
        grid = self.sample_grid.expand(features.shape[0], -1, -1, -1)
        return F.grid_sample(image, grid, mode='bilinear', align_corners=False).reshape(features.shape[0], -1)

    def forward(self, sequence, backbone):
        if len(sequence) < 2: raise ValueError('A causal temporal sequence is required')
        if any(x.shape != sequence[-1].shape for x in sequence): raise ValueError('Aligned feature-map sizes required')
        hexes = [self.receptors(x) for x in sequence]
        state = backbone.init(hexes[0][:, None])
        activity = None
        for previous, current in zip(hexes, hexes[1:]): state, activity = backbone.step(state, previous, current)
        height, width = sequence[-1].shape[-2:]
        key = (height, width)
        if key not in self.raster_indices:
            from scipy.spatial import cKDTree
            yy, xx = np.meshgrid((np.arange(height) + .5) / height, (np.arange(width) + .5) / width, indexing='ij')
            indices = cKDTree(self.points).query(np.stack([xx, yy], -1).reshape(-1, 2))[1]
            self.raster_indices[key] = torch.as_tensor(indices, device=sequence[-1].device)
        raster = activity[:, :, self.raster_indices[key]].reshape(sequence[-1].shape[0], 34, height, width)
        correction = .1 * self.output(raster)
        return sequence[-1] + correction, correction


def detector_features(model, image):
    """Замороженная часть YOLO до его последней головы; без изменения весов."""
    cache = []; x = image
    with torch.no_grad():
        for layer in list(model.model)[:-1]:
            if layer.f != -1: x = cache[layer.f] if isinstance(layer.f, int) else [x if j == -1 else cache[j] for j in layer.f]
            x = layer(x); cache.append(x if layer.i in model.save else None)
    head = model.model[-1]
    return [cache[i] for i in head.f]


def detection_training_predictions(head, features):
    """Та же голова one-to-one, но градиент её входа нужен внешней ветви.

    Штатный Detect.forward намеренно detach-ит эти входы. Веса головы здесь
    заморожены; forward_head сохраняет значения и открывает производную по входу.
    """
    return head.forward_head(features, **(head.one2one if head.end2end else head.one2many))
