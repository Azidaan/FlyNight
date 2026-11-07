"""Тот же выбор рецепторов без повторных маленьких host→CUDA конструкторов."""
import numpy as np
from tracking.b1_eye_layout import CanonicalEyeBank


class CachedHexSampler:
    def __init__(self, centers, minimum_size):
        centers = np.asarray(centers)
        self.centers = centers.astype(np.int64)
        if not np.array_equal(centers, self.centers):
            raise ValueError('Integer receptor locations required')
        self.minimum = tuple(map(int, minimum_size))
        self.indices = {}

    def index_for(self, height, width, device):
        import torch
        if height < self.minimum[0] or width < self.minimum[1]:
            raise ValueError('Runtime eye must already have its original minimum size')
        key = (int(height), int(width), str(device))
        if key not in self.indices:
            points = self.centers + np.array([height // 2, width // 2], np.int64)
            self.indices[key] = torch.from_numpy(points).to(device=device)
        return self.indices[key]

    def __call__(self, sequence):
        h, w = sequence.shape[2:]
        centers = self.index_for(h, w, sequence.device)
        # Это ровно прежний gather и reshape BoxEye.hex_render.
        out = sequence[:, :, centers[:, 0], centers[:, 1]]
        return out.view(*sequence.shape[:2], 1, -1)


class CachedCanonicalEyeBank(CanonicalEyeBank):
    def __init__(self, estimator, *args, **kwargs):
        super().__init__(estimator, *args, **kwargs)
        self.cached_sampler = CachedHexSampler(estimator._local_centers_yx,
            (int(estimator.eye_height), int(estimator.eye_width)))
        # Единственный перенос индексов — до обработки видео, а не в каждом глазе.
        self.cached_sampler.index_for(self.eye_size, self.eye_size, estimator.flyvis.device)
        self.eye.hex_render = self.cached_sampler
