import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from tracking.masa_detic_bdd_input import preprocess_bdd


def test_official_landscape_geometry_rgb_padding():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = [10, 20, 30]
    image, scale = preprocess_bdd(frame)
    assert image.shape == (1, 3, 768, 1344)
    torch.testing.assert_close(scale, torch.tensor([1333/1280, 750/720, 1333/1280, 750/720]), rtol=0, atol=0)
    expected = (torch.tensor([30., 20., 10.]) - torch.tensor([123.675, 116.28, 103.53])) / torch.tensor([58.395, 57.12, 57.375])
    torch.testing.assert_close(image[0, :, 10, 10], expected, rtol=0, atol=0)
    assert torch.count_nonzero(image[:, :, 750:, :]) == 0
    assert torch.count_nonzero(image[:, :, :, 1333:]) == 0


def test_short_side_limit_and_asymmetric_rounding():
    image, scale = preprocess_bdd(np.zeros((101, 100, 3), dtype=np.uint8))
    assert image.shape == (1, 3, 832, 800)
    torch.testing.assert_close(scale, torch.tensor([8., 8., 8., 8.]), rtol=0, atol=0)
    image, scale = preprocess_bdd(np.zeros((719, 1280, 3), dtype=np.uint8))
    assert scale[0] != scale[1]
    assert not torch.cuda.is_initialized()
