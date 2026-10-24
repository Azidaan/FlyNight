import numpy as np
from tracking.z13_specificity_gap import specificity_gap,select


def test_wrong_models_are_read_at_original_peak_not_their_own_maxima():
    maps=np.zeros((4,721));maps[0,3]=8;maps[1:,50]=8
    good=specificity_gap(maps)
    maps[1:,3]=8;maps[1:,50]=0
    assert good>10 and abs(specificity_gap(maps))<1e-12


def test_margin_has_same_constant_offset_invariance_as_existing_readout():
    x=np.random.default_rng(9).normal(size=(4,721))
    assert abs(specificity_gap(x)-specificity_gap(x+np.array([1.,7.,-2.,9.])[:,None]))<1e-12


def test_selection_budget_and_ties_are_identical_without_using_outcomes():
    rows=[dict(label=i,height=1.,gap=1.,outcome='unused') for i in range(101)]
    assert [r['label'] for r in select(list(reversed(rows)),'gap')]==[0,1]
    assert select(rows,'height')==select(rows,'gap')
