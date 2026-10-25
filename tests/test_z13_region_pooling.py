import numpy as np
from tracking.z13_region_pooling import region_probabilities
from tracking.z13_region_readout import region_probabilities as original


def test_mean_is_exact_original_and_support_is_identical():
    args=([5.,-2.,-2.,1.],[[0,0],[1,0],[2,0],[3,0]],[[0,0,2,0],[3,0,3,0],[8,8,9,9]],[True,True,True])
    mean,valid=region_probabilities(*args,reduction='mean');expected,support=original(*args)
    np.testing.assert_array_equal(mean,expected);np.testing.assert_array_equal(valid,support)
    maximum,other=region_probabilities(*args,reduction='max')
    np.testing.assert_array_equal(other,valid);assert maximum[2]==0 and maximum.sum()<1


def test_local_peak_can_survive_averaging_with_same_field_and_regions():
    args=([5.,-3.,-3.,1.,1.],[[0,0],[1,0],[2,0],[3,0],[4,0]],[[0,0,2,0],[3,0,4,0]],[True,True])
    mean,_=region_probabilities(*args,reduction='mean');maximum,_=region_probabilities(*args,reduction='max')
    assert mean[0]<mean[1] and maximum[0]>maximum[1]


def test_constant_field_and_empty_regions_are_finite():
    p,valid=region_probabilities([2.,2.],[[0,0],[1,0]],[[0,0,1,0]],[True])
    np.testing.assert_array_equal(p,[.5]);assert valid.tolist()==[True]
    p,valid=region_probabilities([2.,2.],[[0,0],[1,0]],[],[])
    assert p.shape==valid.shape==(0,)
