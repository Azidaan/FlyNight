import numpy as np
import torch
from tracking.z13_aligned_loss import query_target, query_loss


def test_absent_query_and_observed_extent():
    points=np.array([(x,y) for x in range(5) for y in range(5)],float)
    mask=(points[:,0]>=1)&(points[:,0]<=3)&(points[:,1]>=1)&(points[:,1]<=3)
    target=query_target(mask,points,1,[2,2])
    assert target.argmax()==12 and target[12]==1
    assert np.array_equal(query_target(mask,points,1,None),np.zeros(25))
    # Simultaneous translation of observed geometry and training center preserves labels.
    np.testing.assert_allclose(target,query_target(mask,points+[17,3],1,[19,5]))


def test_easy_negative_scores_are_unpenalized_but_wrong_positive_scores_are_not():
    scores=torch.tensor([-2.,2.],requires_grad=True)
    loss=query_loss(scores,torch.zeros(2)); loss.backward()
    assert scores.grad[0]==0 and scores.grad[1]>0
    at_target=torch.tensor([0.],requires_grad=True)
    query_loss(at_target,torch.ones(1)).backward()
    assert at_target.grad[0]<0
