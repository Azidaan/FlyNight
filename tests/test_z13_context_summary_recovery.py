import copy
import pytest
from scripts.z13_context_summary_recovery_20261020 import selected_video_useful
from scripts.z13_score_swap import selected_keys


def row(label, kind, score, useful=None):
    value = dict(video='v',label=label,kind=kind,p_present=score)
    if useful is not None:
        value['useful']=useful
    return value


def test_unknown_selected_case_consumes_the_same_global_budget_without_imputation():
    rows=[row('unknown',-1,.99),row('target',1,.9,True),row('other',1,.1,False)]
    before=copy.deepcopy(rows)
    chosen=selected_keys(rows,.01)
    assert selected_video_useful(rows,chosen)=={'v':0.0}
    assert rows==before and 'useful' not in rows[0] and rows[0]['kind']==-1


def test_denominator_keeps_all_positive_cases_including_unselected_ones():
    rows=[row('target',1,.99,True),row('other',1,.1,True),row('negative',0,.8,False)]
    assert selected_video_useful(rows,selected_keys(rows,.01))=={'v':.5}


def test_missing_positive_outcome_is_an_error_and_is_not_silently_filled():
    rows=[row('target',1,.9)]
    with pytest.raises(KeyError,match='useful'):
        selected_video_useful(rows,selected_keys(rows,.01))
