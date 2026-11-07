"""Причинные входы Б2, контроль без истории и явное исключение crowd из фона."""
from __future__ import annotations

from collections import deque
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tracking.b2_feature_temporal import TemporalDetectionBranch


class FeatureRing:
    """Только последние пять реально вычисленных query-карт, без GPU-архива."""
    def __init__(self, length=5):
        self.values = deque(maxlen=length)

    def append(self, frame, features):
        if self.values and int(frame) <= self.values[-1][0]:
            raise ValueError('Strictly increasing observed frames required')
        if self.values and features[0].shape != self.values[-1][1][0].shape:
            raise ValueError('Reset required after an image-shape change')
        self.values.append((int(frame), tuple(x.detach() for x in features)))

    def sequence(self):
        return [x[1][0] for x in self.values], [x[0] for x in self.values]

    @property
    def bytes(self):
        return sum(x.numel() * x.element_size() for _, fs in self.values for x in fs)


class CausalP3Branch(TemporalDetectionBranch):
    """Одинаковые адаптеры; last/mean имеют ту же обычную GRU, что контроль B.

    last/mean повторяют агрегат пять раз, поэтому это отдельно обучаемые сильные
    контроли с тем же числом обычных параметров, но не чистая проверка порядка.
    FlyVis получает реальную длительность интервала с шагом не больше 0,02 с.
    """
    def forward(self, sequence, backbone, frames, fps, mode='history'):
        if len(sequence) != 5 or len(frames) != 5:
            raise ValueError('Exactly five observed query frames required')
        if not np.isfinite(fps) or fps <= 0 or any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError('Observed increasing timestamps and real FPS required')
        if any(x.shape != sequence[-1].shape for x in sequence):
            raise ValueError('Same image geometry required')
        if mode == 'last':
            values = [sequence[-1]] * 5
        elif mode == 'mean':
            value = torch.stack(sequence).mean(0)
            values = [value] * 5
        elif mode == 'history':
            values = sequence
        else:
            raise ValueError(mode)
        hexes = [self.receptors(x) for x in values]
        # Fade-in keeps its registered integration step, independently of the
        # unusually short first observed interval.
        if hasattr(backbone, 'dt'):
            backbone.dt = .02
        state = backbone.init(hexes[0][:, None])
        for i, (previous, current) in enumerate(zip(hexes, hexes[1:])):
            if hasattr(backbone, 'steps') and hasattr(backbone, 'dt'):
                duration = (frames[i + 1] - frames[i]) / fps
                backbone.steps = max(1, math.ceil(duration / .02))
                backbone.dt = duration / backbone.steps
            state, activity = backbone.step(state, previous, current)
        height, width = sequence[-1].shape[-2:]
        key = (height, width, str(sequence[-1].device))
        if key not in self.raster_indices:
            from scipy.spatial import cKDTree
            yy, xx = np.meshgrid((np.arange(height) + .5) / height,
                                 (np.arange(width) + .5) / width, indexing='ij')
            indices = cKDTree(self.points).query(np.stack([xx, yy], -1).reshape(-1, 2))[1]
            self.raster_indices[key] = torch.as_tensor(indices, device=sequence[-1].device)
        raster = activity[:, :, self.raster_indices[key]].reshape(sequence[-1].shape[0], 34, height, width)
        correction = .1 * self.output(raster)
        return sequence[-1] + correction, correction


def detection_targets(entry, names, original_shape, input_shape, imgsz=960):
    """GT используется только здесь. Рамки crowd отделены от одиночных целей.

    Возвращает реальные целочисленные отступы LetterBox(auto=True,stride=32).
    Функция не выбирает историю, кадры или признаки по GT.
    """
    h, w = original_shape[:2]
    ih, iw = input_shape[-2:]
    gain = min(imgsz / h, imgsz / w)
    left = round((iw - round(w * gain)) / 2 - .1)
    top = round((ih - round(h * gain)) / 2 - .1)
    mapping = {value: int(key) for key, value in names.items()}
    boxes, classes, crowds, crowd_classes = [], [], [], []
    for obj in entry.get('labels', []):
        if obj.get('category') not in mapping or 'box2d' not in obj:
            continue
        b = obj['box2d']
        box = np.asarray([np.clip(b['x1'], 0, w), np.clip(b['y1'], 0, h),
                          np.clip(b['x2'], 0, w), np.clip(b['y2'], 0, h)], np.float64)
        box = box * gain + [left, top, left, top]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        cls = mapping[obj['category']]
        if obj.get('attributes', {}).get('crowd', False):
            crowds.append(box); crowd_classes.append(cls)
        else:
            x1, y1, x2, y2 = box
            boxes.append([(x1+x2)/2/iw, (y1+y2)/2/ih, (x2-x1)/iw, (y2-y1)/ih])
            classes.append(cls)
    return dict(bboxes=np.asarray(boxes, np.float32).reshape(-1, 4),
                cls=np.asarray(classes, np.int64),
                crowd_xyxy=np.asarray(crowds, np.float32).reshape(-1, 4),
                crowd_cls=np.asarray(crowd_classes, np.int64),
                gain=gain, padding=[left, top])


class CrowdAwareBCE(nn.Module):
    """Не штрафовать фон того же класса внутри crowd; положительная цель важнее.

    Остальные классы и все рамочные потери сохраняют штатный путь YOLO.
    Это заданная политика обучения, а не утверждение об официальном BDD loss.
    """
    def __init__(self, centers, crowd_boxes, crowd_classes, nc):
        super().__init__()
        mask = torch.zeros((centers.shape[0], nc), device=centers.device, dtype=torch.bool)
        for box, cls in zip(crowd_boxes, crowd_classes):
            inside = ((centers[:, 0] >= box[0]) & (centers[:, 0] <= box[2]) &
                      (centers[:, 1] >= box[1]) & (centers[:, 1] <= box[3]))
            mask[:, int(cls)] |= inside
        self.register_buffer('ignore', mask[None])

    def forward(self, scores, targets):
        if scores.shape != targets.shape or scores.shape[0] != 1 or scores.shape[1:] != self.ignore.shape[1:]:
            raise ValueError('One image and matching anchor geometry required')
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
        return loss.masked_fill(self.ignore & (targets <= 0), 0)
