import numpy as np
from tracking.z13_matched_data_budget import matched_batches


def test_reference_size_reproduces_original_epoch_permutations():
    rng=np.random.default_rng(2);expected=[]
    for _ in range(3):
        order=rng.permutation(2919)
        expected.extend(order[k:k+128] for k in range(0,len(order),128))
    actual=list(matched_batches(2919,2919,3,128,2))
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_more_data_has_identical_steps_and_observation_count():
    small=list(matched_batches(2919,2919,3,128,1));large=list(matched_batches(15000,2919,3,128,1))
    assert [len(x) for x in small]==[len(x) for x in large]
    assert sum(map(len,large))==2919*3
    assert max(np.concatenate(large))>2919
    assert min(np.concatenate(large))>=0
