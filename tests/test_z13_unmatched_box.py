import pytest
from tracking.z13_unmatched_box import admit


def test_existing_assignment_is_preserved_by_both_admission_rules():
    assert admit('full',[2,1],0,3)
    assert not admit('unmatched',[2,1],0,3)
    assert not admit('free',[2,1],0,3)


def test_unmatched_rule_and_free_rule_differ_only_for_a_claimed_detection():
    assert admit('unmatched',[-1,2],0,2)
    assert not admit('free',[-1,2],0,2)
    assert admit('free',[-1,2],0,3)
    assert not admit('full',[-1,2],0,-1)
    with pytest.raises(ValueError):admit('other',[-1],0,1)
