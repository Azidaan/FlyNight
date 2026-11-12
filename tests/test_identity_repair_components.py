"""Regression tests for measured ownership/selection failures and generic registration."""
import numpy as np
import pytest
from scipy.spatial import Delaunay
from tracking.census_correspondence import register_translation
from tracking.correspondence_repair_guard import choose_correspondence
from tracking.reference_template_history import terminal_reference_history
from tracking.appearance_recovery_policy import confirm_recovery,observable_episode_tids
from tracking.identity_repair_adapter import compose


def test_registration_recovers_known_translation_without_mutating_images():
    random=np.random.default_rng(731)
    reference=random.integers(0,256,(32,32),dtype=np.uint8)
    current=np.roll(reference,(-2,3),(0,1))
    old_reference=reference.copy();old_current=current.copy()
    mesh=Delaunay(np.array([[0.,0.],[32.,0.],[0.,32.],[32.,32.]]))
    source=np.array([[10.,10.],[16.,10.],[10.,16.],[16.,16.]])
    out=register_translation(reference,current,[10,10,16,16],[0,0,32,32],
                             [13,8,19,14],[0,0,32,32],source,mesh,32)
    assert out['ready'] and out['translation']==[3,-2]
    assert out['cost_tuples'][out['winner']][:2]==(0.,0)
    assert np.array_equal(reference,old_reference) and np.array_equal(current,old_current)


def test_empty_template_never_invents_a_motion():
    image=np.zeros((32,32),np.uint8)
    mesh=Delaunay(np.array([[0.,0.],[32.,0.],[0.,32.],[32.,32.]]))
    out=register_translation(image,image,[-8,-8,-3,-3],[0,0,32,32],
                             [0,0,1,1],[0,0,32,32],np.array([[1.,1.]]),mesh,32)
    assert not out['ready'] and 'translation' not in out


def test_accepted_original_is_preserved_when_universal_repair_would_break_it():
    decision=choose_correspondence(.204126,-.488786,True,False)
    assert decision['accepted'] and not decision['use_repair']
    assert decision['selected_score']==.204126


@pytest.mark.parametrize('support,owner',[(None,False),(False,False),(True,True)])
def test_positive_repair_cannot_bypass_missing_evidence_or_existing_owner(support,owner):
    decision=choose_correspondence(-.398460,.461950,support,owner)
    assert not decision['accepted'] and not decision['use_repair']


def test_explicit_appearance_contradiction_blocks_original_foreign_vote():
    decision=choose_correspondence(.546382,.959217,False,False)
    assert not decision['accepted'] and decision['action']=='defer'


def test_supported_unowned_repair_is_only_a_proposal():
    decision=choose_correspondence(-.398460,.461950,True,False)
    assert decision['accepted'] and decision['use_repair']
    assert decision['action']=='propose_repair'


def test_unknown_original_is_not_silently_replaced():
    assert not choose_correspondence(None,1.,True,False)['accepted']


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_score_cannot_create_an_assignment(bad):
    with pytest.raises(ValueError):choose_correspondence(bad,1.,True,False)


def test_reference_history_uses_only_last_observed_input_and_owns_its_memory():
    original=np.arange(3*721,dtype=np.float32).reshape(3,721)
    before=original.copy()
    result=terminal_reference_history(original)
    assert result.shape==(33,721) and result.dtype==original.dtype
    assert np.array_equal(result,np.repeat(original[-1:],33,axis=0))
    result[0,0]=-1
    assert np.array_equal(original,before)


def test_empty_history_cannot_create_a_reference():
    with pytest.raises(ValueError):terminal_reference_history(np.empty((0,721)))


def recovery_fixture():
    rows=[dict(tid=10,state=2,cls=2,predicted_xyxy=[0,0,10,10]),
          dict(tid=20,state=1,cls=2,predicted_xyxy=[30,0,40,10])]
    candidates=[dict(raw_id=100,box=[0,0,10,10],confidence=.07,cls=2),
                dict(raw_id=101,box=[30,0,40,10],confidence=.8,cls=2)]
    return rows,candidates,np.eye(2),np.eye(2)


def test_complete_mutual_low_proposal_needs_existing_observable_episode():
    rows,candidates,a,b=recovery_fixture()
    closed=confirm_recovery(rows,candidates,a,b,[(20,101)],competition_complete=True)
    opened=confirm_recovery(rows,candidates,a,b,[(20,101)],eligible_tids={10},competition_complete=True)
    assert not closed['proposals']
    assert [(p['tid'],p['raw_id']) for p in opened['proposals']]==[(10,100)]
    assert opened['baseline_pairs']==[[20,101]]


def test_ineligible_competitor_still_prevents_wrong_mutual_recovery():
    rows,candidates,a,b=recovery_fixture()
    a=np.array([[.8,.6],[1.,0.]])
    result=confirm_recovery(rows,candidates,a,b,[],eligible_tids={10},competition_complete=True)
    assert not result['proposals']
    assert result['decisions'][0]['reason']=='not_strict_mutual_nearest_against_all_competitor_rows'


def test_owned_alias_cannot_be_recovered_under_another_id():
    rows,candidates,a,b=recovery_fixture()
    candidates.append(dict(raw_id=102,box=[.1,0,10.1,10],confidence=.06,cls=2))
    b=np.vstack([b,[1.,0.]])
    result=confirm_recovery(rows,candidates,a,b,[(20,102)],eligible_tids={10},competition_complete=True)
    assert not result['proposals']
    assert result['decisions'][0]['reason']=='candidate_alias_owned_by_frozen_native_assignment'


def test_missing_competition_never_becomes_an_assignment():
    rows,candidates,a,b=recovery_fixture()
    result=confirm_recovery(rows,candidates,a,b,[],eligible_tids={10},competition_complete=False)
    assert not result['proposals'] and all(d['action']=='defer' for d in result['decisions'])


def test_both_original_head_outcomes_share_precore_episode_admission():
    proposals=[dict(old_id=10,geometry=[0]*7,reason='head_accept',evidence_valid=False),
               dict(old_id=20,geometry=[0]*7,reason='head_refuse',evidence_valid=False),
               dict(old_id=30,reason='outside_observed_training_gap')]
    assert observable_episode_tids(proposals)=={10,20}


def test_active_row_cannot_start_the_lost_only_branch():
    rows,candidates,a,b=recovery_fixture();rows[0]['state']=1
    result=confirm_recovery(rows,candidates,a,b,[],eligible_tids={10},competition_complete=True)
    assert not result['proposals']


def test_production_adapter_preserves_a_correct_owner_against_positive_old_branch():
    rows,candidates,a,b=recovery_fixture()
    rows[1]['predicted_xyxy']=[0,0,10,10]
    a=np.array([[1.,0.],[1.,0.]])
    snapshot=dict(video='fixture',frame=6,target_row=0,target_column=0,target_tid=10,
                  target_raw_id=100,high_raw_ids=[100,101],actual_first=[[1,0]],
                  assigned={'20':100},base_cost=[[.9,.9],[.1,.9]],corrected_cost=[[.9,.9],[.1,.9]],
                  native_proposals=[dict(old_id=10,geometry=[0]*7,reason='head_refuse')])
    edits,note=compose(snapshot,rows,candidates,a,b,-.4,.5,'guarded_census_appearance_low_native_domain')
    assert edits['freeze']=='all' and note['appearance']['owned_by_other']
    assert not edits['append_columns'] and not note['high_choice']['accepted']
