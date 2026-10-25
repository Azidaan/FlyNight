import numpy as np
import pytest
from tracking.z13_action_feasibility import conditional_net, required_recovery, helpful_box_mask


def test_known_population_balance_and_recovery_requirement():
    # 5 helpful opportunities among100; recover4, damage1 of the other95.
    assert conditional_net(.05, .8, 1 / 95) == pytest.approx(.03)
    need = required_recovery(.05, .02, 1 / 95)
    assert need == pytest.approx(.6)
    assert conditional_net(.05, need, 1 / 95) == pytest.approx(.02)


def test_ceiling_and_infeasible_target():
    np.testing.assert_array_equal(conditional_net([0, .01, .1], 1, 0), [0, .01, .1])
    assert required_recovery(.01, .02, 0) > 1
    with pytest.raises(ValueError):
        conditional_net(.1, 1.01, 0)


def test_null_is_not_a_box_and_bindings_are_checked():
    assert not helpful_box_mask([-1, 2], [True, False])
    assert helpful_box_mask([-1, 2], [False, True])
    with pytest.raises(ValueError):
        helpful_box_mask([-1, 2], [True])
