from scripts.z13_refusal_causes_20261025 import bucket, curve_stats


def test_no_proposal_and_missed_rank_are_separate():
    no = {'beneficial':False}
    yes = {'beneficial':True}
    assert bucket(False,[no,no,no]) == 'no_current_candidate_can_help'
    assert bucket(False,[no,yes,no]) == 'only_other_candidate_can_help'
    assert bucket(False,[yes,no,no]) == 'top1_can_help'
    assert bucket(True,[no,no,no]) == 'baseline_already_correct'


def test_ranking_does_not_fit_or_choose_a_threshold():
    r = curve_stats([.1,.9,.2,.8],[False,True,False,True])
    assert r['roc_auc'] == r['average_precision'] == 1.
    assert r['prevalence'] == .5 and r['n'] == 4
    assert 'threshold' not in r


def test_single_class_does_not_invent_an_auc():
    r = curve_stats([.1,.2],[False,False])
    assert r['roc_auc'] is None and r['average_precision'] is None
