"""Assignment gates are constraints, not a filter after optimization."""
from itertools import combinations, permutations, product
from decimal import Decimal

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from tracking.flyvis_tracker import (
    FlyVisTracker, FlyVisTrackerConfig, MotionField, WeightedGeometryAssignment,
)


class MatrixAssignment(WeightedGeometryAssignment):
    def __init__(self, costs, valid=None):
        self.costs = np.asarray(costs, dtype=np.float64)
        self.valid = np.isfinite(self.costs) if valid is None else np.asarray(valid, bool)

    def cost(self, *args):
        return self.costs.copy(), self.valid.copy()

    def solve(self, limit):
        rows, cols = self.costs.shape
        cfg = FlyVisTrackerConfig(max_assignment_cost=limit)
        return self.assign([None] * rows, np.empty((rows, 4)), np.empty((cols, 4)),
                           np.zeros(cols, int), cfg)


def test_real_boxes_do_not_lose_valid_identity_to_over_threshold_assignments():
    config = FlyVisTrackerConfig(iou_weight=0., motion_weight=1., size_weight=0.,
                                 max_assignment_cost=.15)
    tracker = FlyVisTracker(config)
    confidence, classes = np.full(2, .9), np.full(2, 2)
    tracker.update(np.array([[0, 0, 10, 10], [3, 0, 13, 10]]), confidence,
                   classes, motion_field=MotionField.empty())
    output = tracker.update(np.array([[1, 0, 11, 10], [-2, 0, 8, 10]]), confidence,
                            classes, motion_field=MotionField.empty())
    assert output[:, 4].tolist() == [1, 3]
    np.testing.assert_array_equal(output[0, :4], [1, 0, 11, 10])
    assert tracker.events[-2]["event"] == "lost"


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (0, 3), (3, 0)])
def test_all_invalid_rectangular_assignments_leave_everything_unmatched(shape):
    matches, rows, cols = MatrixAssignment(np.full(shape, 9.)).solve(2.)
    assert matches == [] and rows == set(range(shape[0])) and cols == set(range(shape[1]))


def possible_assignments(shape):
    rows, cols = shape
    return [tuple(zip(rr, cc))
            for size in range(min(shape) + 1)
            for rr in combinations(range(rows), size)
            for cc in permutations(range(cols), size)]


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 2), (2, 3), (3, 2)])
def test_exhaustive_small_assignment_maximizes_cardinality_then_minimizes_cost(shape):
    candidates = possible_assignments(shape)
    for entries in product((-2., 0., 1., 3., np.inf), repeat=int(np.prod(shape))):
        cost = np.asarray(entries).reshape(shape)
        permitted = np.isfinite(cost) & (cost <= 1.5)
        possible = [pairs for pairs in candidates if all(permitted[r, c] for r, c in pairs)]
        expected_count = max(map(len, possible))
        expected_cost = min(sum(cost[r, c] for r, c in pairs)
                            for pairs in possible if len(pairs) == expected_count)
        matches, unmatched_rows, unmatched_cols = MatrixAssignment(cost).solve(1.5)
        assert len(matches) == expected_count
        assert sum(cost[r, c] for r, c in matches) == expected_cost
        assert unmatched_rows == set(range(shape[0])) - {r for r, _ in matches}
        assert unmatched_cols == set(range(shape[1])) - {c for _, c in matches}


@pytest.mark.parametrize("cost", [
    [[-2e307, 0., np.inf], [np.inf, 2e307, -2e307]],
    [[-1e308, 5e307], [5e307, np.inf]],
    [[1e308, 1e308, np.inf], [np.inf, -1e308, 0.]],
    [[-1e8, 0.], [0., np.inf]],
])
def test_large_and_negative_costs_cannot_outweigh_valid_cardinality(cost):
    policy = MatrixAssignment(cost)
    matches, _, _ = policy.solve(float("inf"))
    candidates = [pairs for pairs in possible_assignments(policy.costs.shape)
                  if all(policy.valid[r, c] for r, c in pairs)]
    count = max(map(len, candidates))
    total = lambda pairs: sum((Decimal(str(policy.costs[r, c])) for r, c in pairs), Decimal(0))
    expected = min(total(pairs) for pairs in candidates if len(pairs) == count)
    assert len(matches) == count and total(matches) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_cost_marked_valid_is_rejected(value):
    with pytest.raises(ValueError, match="стоимост"):
        MatrixAssignment([[value]], [[True]]).solve(float("inf"))


def test_ordinary_unlimited_threshold_keeps_previous_solver_choices_including_ties():
    rng = np.random.default_rng(911)
    for shape in ((2, 3), (3, 2), (4, 4)):
        for _ in range(100):
            costs = rng.integers(0, 4, shape).astype(np.float32)
            valid = rng.random(shape) > .2
            old_rows, old_cols = linear_sum_assignment(np.where(valid, costs, 1e6))
            old = [(r, c) for r, c in zip(old_rows.tolist(), old_cols.tolist()) if valid[r, c]]
            # The production cost() already marks geometric exclusions inf.
            matches, _, _ = MatrixAssignment(np.where(valid, costs, np.inf)).solve(float("inf"))
            assert matches == old
