import numpy as np
from tracking.z13_query_memory import fuse,expert_maps
from tracking.z13_target_center import ridge_scores,observed_center_target


def test_zero_weight_and_single_expert_preserve_baseline_exactly():
    a=np.arange(15,dtype=float).reshape(3,5)
    for mode in ('fixed','query'):
        np.testing.assert_array_equal(fuse(a,mode,0,.25),a[-1])
        np.testing.assert_array_equal(fuse(a[-1:],mode,1,.25),a[-1])


def test_query_reads_different_expert_at_different_locations():
    maps=np.array([[4.,0,0,0],[0,0,0,4.]])
    q=fuse(maps,'query',1,.1);fixed=fuse(maps,'fixed',1)
    assert q[0]>fixed[0]+1.9 and q[-1]>fixed[-1]+1.9
    assert q[1]==q[2]==0


def test_padded_observation_dedup_and_last_model_match_original():
    rng=np.random.default_rng(3);obs=rng.normal(size=(4,7,2));ms=np.tile([0,0,1,1,1,0,0],(4,1));pts=np.stack([np.arange(7),np.zeros(7)],1);xs=rng.normal(size=(2,7,2))
    maps,slots=expert_maps(obs,ms,xs,[5,5,5,5],pts,1.,.1)
    assert slots==[3]
    np.testing.assert_array_equal(maps[-1],ridge_scores(obs[3],observed_center_target(ms[3],pts,1.),xs,[.1])[0])
