import copy
import pytest
from tracking.bbox_singleton_gate import proposals


def event(candidates):
    return dict(frame=10,rows=[dict(old_tid=3,age=2,row_data=dict(history=[dict(frame=8,box=[0,0,10,10])]),candidates=candidates)])


def candidate(tid=8,iou=.7,box=None,birth=None):
    return dict(new_tid=tid,forward_iou=iou,box=box or [2,0,12,10],birth=dict(frame=9,box=birth or [1,0,11,10]),cost=.9,column=1)


def test_one_candidate_is_eligible_without_two_box_or_retinal_requirements():
    assert len(proposals(event([candidate()])))==1


def test_any_number_of_irrelevant_boxes_does_not_prevent_a_singleton():
    assert len(proposals(event([candidate(),candidate(None,.1),candidate(None,.05)])))==1


def test_unconfirmed_competing_free_box_still_prevents_a_singleton():
    assert not proposals(event([candidate(),candidate(None,.6)]))


def test_new_native_public_confirmation_is_required():
    assert not proposals(event([candidate(None)]))


def test_backward_veto_does_not_turn_two_forward_candidates_into_one():
    assert not proposals(event([candidate(),candidate(9,box=[100,0,110,10])]),bidirectional=True)


def test_backward_translation_and_stale_rejection_do_not_mutate_input():
    e=event([candidate()]);before=copy.deepcopy(e)
    assert proposals(e,bidirectional=True)[0]['backward_iou']==pytest.approx(1.)
    assert not proposals(e,cap=1)
    assert e==before


def test_invalid_geometry_settings_fail():
    with pytest.raises(ValueError):proposals(event([candidate()]),threshold=0)
