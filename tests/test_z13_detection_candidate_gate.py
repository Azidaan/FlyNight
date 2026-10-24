import numpy as np
from scripts.z13_detection_candidate_gate_20261024 import candidates


def test_selection_is_translation_equivariant():
    boxes=np.array([[1.,1.,3.,3.],[4.,4.,6.,6.],[7.,7.,9.,9.],[2.,7.,4.,9.]])
    scores=np.array([.9,.8,.7,.6]);pred=np.array([3.,3.]);crop=np.array([0.,0.,10.,10.])
    a=candidates(boxes,scores,pred,crop)
    shift=np.array([120.,-37.]);b=candidates(boxes+np.tile(shift,2),scores,pred+shift,crop+np.tile(shift,2))
    assert np.array_equal(a[0],b[0])
    assert np.allclose(a[1]+shift,b[1]) and np.allclose(a[2]+shift,b[2])


def test_no_proposal_is_not_fabricated():
    ids,centers,allcenters=candidates([[0,0,2,2]],[.1],[1,1],[0,0,3,3])
    assert len(ids)==0 and centers.shape==allcenters.shape==(0,2)


def test_outside_and_low_score_boxes_do_not_displace_valid_candidates():
    boxes=[[1,1,3,3],[2,2,4,4],[3,3,5,5],[100,100,102,102],[0,0,2,2]]
    ids,centers,_=candidates(boxes,[.9,.8,.7,1.,.24],[0,0],[0,0,10,10])
    assert ids.tolist()==[0,1,2] and np.all(np.isfinite(centers))
