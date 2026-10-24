import numpy as np
import pytest
from tracking.z13_presence_scope import label_reasons, select_known


def test_unknown_identity_is_not_a_negative_example():
    r = label_reasons([-1, 0, 0, 0, 1, -1],
        [[np.nan]*2, [np.nan]*2, [11, 5], [1, 1], [5, 5], [np.nan]*2],
        [[0, 0, 10, 10]]*6, ['', 'a', 'a', 'a', 'a', 'a'], [0, 0, 0, 0, 0, 0])
    assert r == ['unknown_observed_identity_or_stale', 'target_not_annotated_in_query',
        'annotated_target_outside_square', 'annotated_target_outside_hex_field',
        'target_center_in_eye', 'unknown_query_annotation_gap']
    with pytest.raises(ValueError):
        label_reasons([0], [[np.nan]*2], [[0, 0, 10, 10]], [''], [0])


def test_oracle_preserves_within_presence_order_and_ignores_outcomes():
    rows = [dict(label=i, kind=k, score=s, outcome='harmful')
        for i, k, s in [(3, 1, 2), (1, 1, 2), (2, 0, 1000), (4, -1, 1e9)]]
    before = [r.copy() for r in rows]
    assert [r['label'] for r in select_known(rows, True, .5)] == [1, 3]
    assert [r['label'] for r in select_known(rows, False, .5)] == [2, 1]
    for r in rows:
        r['outcome'] = 'beneficial'
    assert [r['label'] for r in select_known(rows, True, .5)] == [1, 3]
    assert all(a['score'] == b['score'] for a, b in zip(rows, before))


def test_no_unknown_is_silently_given_absence_label():
    assert select_known([dict(label=0, kind=-1, score=100, outcome='unknown')], True) == []
