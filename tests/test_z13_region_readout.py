import numpy as np
from tracking.z13_region_readout import region_probabilities, cost_row


def test_zero_calibration_preserves_original_cost_exactly():
    old = np.asarray([.123456789012345, .9])
    assert np.array_equal(cost_row(old, [.1,.9], [.7,.8], [True,True], 0), old)


def test_out_of_field_region_keeps_baseline_and_null_exists():
    p, valid = region_probabilities([1.,0.,-1.], [[0,0],[1,0],[2,0]], [[0,0,1,0],[5,5,6,6]], [True,True])
    assert valid.tolist() == [True,False] and p[1] == 0 and 0 < p.sum() < 1
    assert cost_row([.3,.4], p, [.9,.8], valid, 1)[1] == .4


def test_changing_response_changes_region_choice_with_fixed_geometry():
    pts = [[0,0],[1,0],[2,0],[3,0]]
    boxes = [[0,0,1,0],[2,0,3,0]]
    first, _ = region_probabilities([1.,1.,-1.,-1.], pts, boxes, [True,True])
    second, _ = region_probabilities([-1.,-1.,1.,1.], pts, boxes, [True,True])
    assert first[0] > first[1] and second[1] > second[0]
