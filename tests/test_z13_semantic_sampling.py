import numpy as np
from tracking.z13_semantic_sampling import semantic_negatives,semantic_batches


def test_only_unambiguous_other_same_class_negative():
    label=dict(kind=1,gt=[0,0],pred=[3,3],scale=4)
    objects=[('a','car',[-2,-2,2,2]),('b','car',[9,9,11,11]),('c','bus',[19,19,21,21])]
    np.testing.assert_array_equal(semantic_negatives([[0,0],[10,10],[20,20]],label,'a','car',objects),[False,True,False])
    objects.append(('d','car',[9,9,11,11]))
    assert not semantic_negatives([[10,10]],label,'a','car',objects)[0]


def test_unknown_and_old_positive_never_reclassified():
    objects=[('b','car',[0,0,2,2])]
    assert not semantic_negatives([[1,1]],{'kind':-1},'a','car',objects)[0]
    label=dict(kind=1,gt=[0,0],pred=[9,9],scale=10)
    assert not semantic_negatives([[1,1]],label,'a','car',objects)[0]


def test_equal_budget_determinism_and_hard_presence():
    batches=list(semantic_batches(20778,2919,2,[3,5,8]))
    assert len(batches)==1150 and sum(map(len,batches))==145950
    assert all(np.isin(b,[3,5,8]).sum()>=len(b)//2 for b in batches)
    for a,b in zip(batches,semantic_batches(20778,2919,2,[3,5,8])):np.testing.assert_array_equal(a,b)
