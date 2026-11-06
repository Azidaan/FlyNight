"""A rejected frame must not contaminate later temporal adaptation."""
import numpy as np
import pytest

from tracking.retina import Retina, RetinaConfig


@pytest.mark.parametrize("mode", ["photoreceptor", "lamina"])
@pytest.mark.parametrize("tau_filter", [0., .1])
@pytest.mark.parametrize("changed_shape", [(1, 32), (16, 1), (8, 32)])
def test_stateful_retina_rejects_shape_change_before_updating_state(mode, tau_filter, changed_shape):
    config = RetinaConfig(mode=mode, tau_filter=tau_filter)
    retina, control = Retina(config, 30.), Retina(config, 30.)
    first = np.arange(16 * 32, dtype=np.uint16).reshape(16, 32).astype(np.uint8)
    retina.process(first)
    control.process(first)
    with pytest.raises(ValueError, match="reset"):
        retina.process(np.full(changed_shape, 90, np.uint8))
    following = np.full((16, 32), 60, np.uint8)
    np.testing.assert_array_equal(retina.process(following), control.process(following))
    retina.reset()
    new_video = np.full(changed_shape, 90, np.uint8)
    np.testing.assert_array_equal(retina.process(new_video), Retina(config, 30.).process(new_video))


@pytest.mark.parametrize("mode", ["none", "norm", "photoreceptor", "lamina"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bad_frame_is_rejected_and_next_clean_frame_matches_control(mode, value):
    config = RetinaConfig(mode=mode, tau_filter=.1)
    retina, control = Retina(config, 30.), Retina(config, 30.)
    first = np.full((16, 32), 20, np.uint8)
    retina.process(first)
    control.process(first)
    bad = np.full((16, 32), 40., np.float32)
    bad[4, 5] = value
    with pytest.raises(ValueError, match="NaN|Inf"):
        retina.process(bad)
    clean = np.full((16, 32), 60, np.uint8)
    np.testing.assert_array_equal(retina.process(clean), control.process(clean))


@pytest.mark.parametrize("shape", [(0, 32), (16, 0), (16, 32, 3), (32,)])
def test_retina_requires_nonempty_grayscale_frame(shape):
    with pytest.raises(ValueError, match="HxW"):
        Retina(RetinaConfig(), 30.).process(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("mode", ["none", "norm"])
def test_stateless_modes_accept_new_frame_sizes(mode):
    retina = Retina(RetinaConfig(mode=mode), 30.)
    retina.process(np.zeros((16, 32), np.uint8))
    assert retina.process(np.zeros((1, 32), np.uint8)).shape == (1, 32)
