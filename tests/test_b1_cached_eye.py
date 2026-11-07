import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import ast
from pathlib import Path
import numpy as np
import torch
import pytest
from types import SimpleNamespace
from tracking.b1_cached_eye import CachedHexSampler


def official_reference():
    root = Path(__file__).resolve().parents[1]
    path = root / 'tools/flyvis-runtime/flyvis/datasets/rendering/eye.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoxEye')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'hex_render')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['hex_render']


@pytest.mark.parametrize('shape', [(9, 9), (12, 10), (17, 19)])
def test_cached_gather_equals_actual_official_function(shape):
    centers = np.array([[-2, -2], [0, 0], [2, 1]], np.int64)
    eye = SimpleNamespace(min_frame_size=torch.tensor([9, 9]), receptor_centers=torch.from_numpy(centers))
    sequence = torch.arange(2*3*shape[0]*shape[1], dtype=torch.float32).reshape(2, 3, *shape)
    expected = official_reference()(eye, sequence)
    sampler = CachedHexSampler(centers, (9, 9))
    first = sampler(sequence)
    index = sampler.index_for(*shape, sequence.device)
    second = sampler(sequence)
    assert torch.equal(first, expected) and torch.equal(second, expected)
    assert sampler.index_for(*shape, sequence.device) is index


def test_wrong_size_is_rejected_instead_of_changing_sampling():
    sampler = CachedHexSampler([[0, 0]], (9, 9))
    with pytest.raises(ValueError): sampler(torch.zeros(1, 1, 8, 9))
