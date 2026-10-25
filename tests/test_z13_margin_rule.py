import numpy as np
from tracking.z13_margin_rule import actions, crossfit, outcomes


def test_margin_invariant_to_additional_inferior_candidates():
    scores = [[.4], [.4, 0, 0, 0, 0, 0]]
    assert actions(scores, 'margin', .2).tolist() == [0, 0]
    assert actions(scores, 'probability', .4).tolist() == [0, -1]


def test_no_action_and_unknown_are_not_benefit():
    good = np.ones((3, 1), bool); harm = np.zeros((3, 1), bool)
    assert outcomes([-1, 0, 0], good, harm, [1, -1, 1]).tolist() == [0, 0, 1]
    assert actions([[0], [-.1], [2]], 'margin', None).tolist() == [-1, -1, -1]


def test_validation_labels_cannot_select_own_threshold():
    logits = [[.1], [.8], [.1], [.8]]
    good = np.array([[1], [0], [0], [1]], bool); harm = ~good
    args = (np.ones(4), np.array([0, 0, 1, 1]), ['a', 'b'], {'a': 0, 'b': 1}, 'margin')
    first, _ = crossfit(logits, good, harm, *args)
    changed_good = good.copy(); changed_harm = harm.copy()
    changed_good[:2] = ~good[:2]; changed_harm[:2] = ~harm[:2]
    second, _ = crossfit(logits, changed_good, changed_harm, *args)
    assert first['folds'][0]['threshold'] == second['folds'][0]['threshold']
