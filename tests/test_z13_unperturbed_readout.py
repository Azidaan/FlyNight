from scripts.z13_unperturbed_readout_20261024 import select_rows,measure
import numpy as np


def test_subsets_hold_video_kind_counts_without_training_on_perturbed_clean_rows():
    labels=[dict(video=v,kind=k,episode=e,pos=0) for v in ('a','b') for k in (0,1) for e in range(4)]
    flags=np.array([False,True,False,True]*4)
    indices,details=select_rows(labels,flags)
    assert len(indices['clean'])==len(indices['mixed'])==8
    assert not flags[indices['clean']].any()
    assert all(d['clean']==d['mixed']==2 for d in details.values())


def test_unknown_outcomes_are_bounds_not_fabricated_errors():
    rows=[dict(video='a',kind=-1,p_present=2.,useful=False),dict(video='a',kind=1,p_present=1.,useful=True,err=0,err_pred=1)]
    y,c,d=measure(rows)
    assert y['a']==0 and c['a']==1
    assert d['a']['known_precision'] is None and d['a']['possible_yield_upper']==1


def test_selection_does_not_inspect_outcomes():
    rows=[dict(video='a',kind=1,p_present=2.,useful=False,err=2,err_pred=1),dict(video='a',kind=1,p_present=1.,useful=True,err=0,err_pred=1)]
    y,_,d=measure(rows)
    assert y['a']==0 and d['a']['strict_center_worse']==1
