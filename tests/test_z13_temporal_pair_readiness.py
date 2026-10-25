import numpy as np
from scripts.z13_temporal_pair_readiness_20261025 import choose_previous, pair_labels


def test_previous_is_strictly_causal_and_last_duplicate():
    frames = np.array([2, 2, 5, 6, 9])
    assert choose_previous(range(5), frames, 3, lambda i: True, {2, 5, 6, 9}) == 2
    assert choose_previous(range(5), frames, 2, lambda i: True, {2, 5, 6, 9}) == 1


def test_previous_respects_episode_prefix_support_and_real_annotation():
    frames = np.array([1, 3, 5, 8, 9])
    assert choose_previous(range(2, 5), frames, 4, lambda i: i != 3, {1, 3, 5, 9}) == 2
    assert choose_previous(range(3, 5), frames, 4, lambda i: True, {1, 3, 5, 9}) is None


def test_unknown_detection_is_not_negative_correspondence():
    assert pair_labels(['a', None, 'b'], ['b', None]) == [[0, 0, 0], [2, 0, 1]]


def test_different_known_identities_are_negative_even_with_same_tracker_group():
    assert pair_labels(['target'], ['neighbor']) == [[0, 0, 0]]
