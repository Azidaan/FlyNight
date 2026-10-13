"""Scientific-contract tests for spatial matching and confidence, CPU only."""
import numpy as np
import torch

from tracking.z13_spatial import make_spatial_readout, quality_features
from tracking.z13_readout import HexGeometry, center_estimate, to_frame
from scripts.z13_threshold_diagnostic import exact_curve


def geometry():
    from scipy.spatial import cKDTree
    # 721 points are an API requirement, not an image benchmark.
    points = np.stack(np.meshgrid(np.arange(31), np.arange(24)), -1).reshape(-1, 2)[:721].astype(np.float32)
    points = points * 3 + 5
    geom = HexGeometry(points, 100)
    return geom


def test_spatial_readout_preserves_position_and_ignores_padding():
    torch.manual_seed(0)
    geom = geometry()
    ro = make_spatial_readout(torch)(34, geom.points, 100, geom.neighbors)
    feat = torch.randn(2, 34, 721, requires_grad=True)
    mask = torch.zeros(2, 721, dtype=torch.bool); mask[:, 30:55] = True
    ex, valid = ro.exemplar(feat, mask)
    search = torch.randn(2, 34, 721)
    first = ro.logits((ex, valid), search)
    altered = ex.clone(); altered[~valid] = 1e6
    assert torch.equal(first, ro.logits((altered, valid), search))
    changed = feat.detach().clone(); changed[:, :, 30:55] = changed[:, :, 30:55].flip(-1)
    second = ro.logits(ro.exemplar(changed, mask), search)
    assert not torch.allclose(first, second)
    first.sum().backward()
    assert torch.isfinite(feat.grad).all() and feat.grad.abs().sum() > 0


def test_observable_quality_features_and_empty_mask():
    import pytest
    geom = geometry()
    p = np.ones(722) / 722
    xy = center_estimate(p[:721], geom)
    crop = [30, 20, 130, 120]
    center = to_frame(xy, crop, 100)
    x = quality_features(p, geom, xy, center, center, crop)
    assert x.shape == (12,) and np.isfinite(x).all()
    assert x[8] == x[9] == x[10] == 0
    ro = make_spatial_readout(torch)(34, geom.points, 100, geom.neighbors)
    with pytest.raises(ValueError, match='Empty'):
        ro.exemplar(torch.zeros(1, 34, 721), torch.zeros(1, 721, dtype=torch.bool))


def test_exact_threshold_does_not_merge_close_scores_or_split_ties():
    good = {'kind': 1, 'err': 0., 'err_pred': 1., 'scale': 2., 'p_present': .700049}
    bad = {'kind': 0, 'p_present': .700011}
    result = exact_curve([dict(good) for _ in range(50)] + [dict(bad) for _ in range(7)])
    assert result['tau'] == .700049 and result['accepted']['precision'] == 1
    result = exact_curve([dict(good) for _ in range(50)] + [dict(bad, p_present=.700049) for _ in range(7)])
    assert result['tau'] is None
