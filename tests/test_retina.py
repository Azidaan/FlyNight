"""Фронт-энд «фоторецептор + ламина» (tracking/retina.py, Н9/М1)."""

import numpy as np
import pytest

from tracking.retina import MODES, Retina, RetinaConfig, make_retina


def _scene(mean: float, contrast: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, size=(72, 128)).astype(np.float32)
    return np.clip(mean + contrast * base, 0.0, 1.0)


def test_none_is_identity():
    gray = (np.arange(0, 256, dtype=np.float32).reshape(16, 16)).astype(np.uint8)
    assert np.allclose(make_retina("none", 30.0).process(gray), gray.astype(np.float32))


@pytest.mark.parametrize("mode", [m for m in MODES if m != "none"])
def test_output_range_and_shape(mode):
    retina = make_retina(mode, 30.0)
    out = retina.process((_scene(0.05, 0.02) * 255).astype(np.uint8))
    assert out.shape == (72, 128) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_photoreceptor_equalizes_dark_and_bright_scenes():
    """Одна и та же сцена при яркости 0,05 и 0,5 после адаптации должна давать близкий размах."""
    dark = (_scene(0.05, 0.03) * 255).astype(np.uint8)
    bright = (_scene(0.5, 0.3) * 255).astype(np.uint8)
    raw_ratio = bright.astype(np.float32).std() / dark.astype(np.float32).std()
    r = make_retina("photoreceptor", 30.0)
    out_dark = [r.process(dark) for _ in range(20)][-1]
    r.reset()
    out_bright = [r.process(bright) for _ in range(20)][-1]
    adapted_ratio = out_bright.std() / out_dark.std()
    assert raw_ratio > 5.0
    assert adapted_ratio < 2.0
    assert abs(out_dark.mean() / 255.0 - 0.45) < 0.1 and abs(out_bright.mean() / 255.0 - 0.45) < 0.1


def test_adaptation_follows_time_constant():
    r = Retina(RetinaConfig(mode="photoreceptor", tau_adapt=0.5), fps=30.0)
    dark = np.full((32, 32), 10, np.uint8)
    for _ in range(30):
        r.process(dark)
    first = r.process(np.full((32, 32), 100, np.uint8)).mean()
    later = None
    for _ in range(60):
        later = r.process(np.full((32, 32), 100, np.uint8)).mean()
    assert first > later  # скачок яркости сначала виден, затем адаптация возвращает к среднему
    assert abs(later / 255.0 - 0.45) < 0.05


def test_lamina_removes_smooth_gradient():
    x = np.linspace(0.2, 0.8, 640, dtype=np.float32)
    gradient = np.repeat(x[None, :], 240, axis=0)
    r = make_retina("lamina", 30.0)
    out = None
    for _ in range(10):
        out = r.process((gradient * 255).astype(np.uint8))
    interior = out[40:-40, 120:-120]  # вдали от краёв (размытия σ = 24 и 12 px)
    assert interior.std() < 6.0 and abs(interior.mean() / 255.0 - 0.45) < 0.05  # плавный градиент почти убран


def test_invalid_mode():
    with pytest.raises(ValueError):
        RetinaConfig(mode="unknown")
