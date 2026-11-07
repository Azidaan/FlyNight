"""BDD reference input contract, separate from historical 1024 preparation.

Source: MASA c5472b9c config datasets/bdd/bdd_dataset.py: Resize(1333,800),
keep_ratio; TrackDataPreprocessor RGB/ImageNet/pad32 from BDD model config.
No claim of complete official feature-path numerical parity.
"""
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_bdd(frame_bgr, device='cpu'):
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3
    h, w = frame_bgr.shape[:2]
    factor = min(1333.0 / max(h, w), 800.0 / min(h, w))
    # MMCV rescale_size rounds positive dimensions with int(x + .5).
    new_w, new_h = int(w * factor + .5), int(h * factor + .5)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1])).permute(2, 0, 1).to(device).float()
    mean = image.new_tensor([123.675, 116.28, 103.53])[:, None, None]
    std = image.new_tensor([58.395, 57.12, 57.375])[:, None, None]
    image = (image - mean) / std
    pad_h, pad_w = math.ceil(new_h / 32) * 32, math.ceil(new_w / 32) * 32
    image = F.pad(image, (0, pad_w-new_w, 0, pad_h-new_h))[None]
    # Rounded dimensions produce two actual coordinate scales; preserve both.
    scale = image.new_tensor([new_w / w, new_h / h, new_w / w, new_h / h])
    return image, scale
