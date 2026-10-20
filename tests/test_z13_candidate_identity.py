import numpy as np
from tracking.z13_candidate_identity import IdentityReadout


def test_appearance_preserves_identity_when_geometry_prefers_distractor():
    a=IdentityReadout([0,0],[1,0],1.,.8);g=IdentityReadout([0,0],[1,0],0.,.8)
    for r in (a,g):r.step([[0,0],[4,0]],[[1,0],[0,1]],10)
    assert a.step([[3,0],[1,0]],[[1,0],[0,1]],10)[0]==0
    assert g.step([[3,0],[1,0]],[[1,0],[0,1]],10)[0]==1


def test_unmatched_target_can_be_redetected_without_relabeling_distractor():
    r=IdentityReadout([0,0],[1,0],1.,.2)
    r.step([[0,0],[4,0]],[[1,0],[0,1]],10)
    assert r.step([[4,0]],[[0,1]],10)[0]==-1
    assert r.step([[4,0],[1,0]],[[0,1],[1,0]],10)[0]==1


def test_no_future_information_and_bounded_confidence():
    def prefix():
        r=IdentityReadout([0,0],[1,0],.5,.5)
        return r,[r.step([[0,0]],[[1,0]],10),r.step([[1,0]],[[1,0]],10)]
    r,a=prefix();r.step([[100,100]],[[-1,0]],10);_,b=prefix()
    assert a==b
    assert all(0<=p<=1 for _,p in a)
