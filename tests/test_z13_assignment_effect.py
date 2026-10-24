import numpy as np
from scripts.z13_assignment_effect_20261024 import assignment_vector,action_outcome


def test_assignment_accounts_for_competing_rows():
    c=np.array([[.2,.9],[.1,.3]],dtype=np.float32)
    before=assignment_vector(c,.8)
    assert before.tolist()==[0,1]
    c[0]=[.9,.2];after=assignment_vector(c,.8)
    assert after.tolist()==[1,0]


def test_stealing_a_correct_match_is_not_a_net_benefit():
    r=action_outcome([False,True],[True,False],0)
    assert not r['beneficial'] and r['correct_count_delta']==0


def test_own_correct_match_must_be_preserved_even_if_other_rows_improve():
    r=action_outcome([True,False,False],[False,True,True],0)
    assert r['harmful'] and not r['beneficial']
