import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from tracking.z13_anchor_state import anchor_features

class TensorOps:
    @staticmethod
    def cat(xs,dim):return np.concatenate(xs,axis=dim)

def recurrent(_,arm,bb,x,nb,needed):
    # Nonlinear state makes both dropping the exemplar and retaining middle
    # search frames observable, unlike a memoryless test producer.
    state=np.zeros_like(x[:,0]);out={}
    for t in range(x.shape[1]):
        state=np.tanh(.7*state+x[:,t])
        if t in needed:out[t]=state.copy()
    return out

def test_first_search_and_exemplar_match_original_recurrence():
    x=np.random.default_rng(5).normal(size=(2,9,7));before=x.copy()
    out=anchor_features(TensorOps,'D',None,x,None,{3,4,8},recurrent)
    old=recurrent(None,'D',None,x,None,{3,4,8})
    np.testing.assert_array_equal(out[3],old[3]);np.testing.assert_array_equal(out[4],old[4])
    assert not np.allclose(out[8],old[8]);np.testing.assert_array_equal(x,before)

def test_middle_search_history_cannot_change_later_query():
    x=np.random.default_rng(8).normal(size=(2,10,7));other=x.copy();other[:,4:9]+=9
    a=anchor_features(TensorOps,'R1',None,x,None,{9},recurrent)[9]
    b=anchor_features(TensorOps,'R1',None,other,None,{9},recurrent)[9]
    np.testing.assert_array_equal(a,b)
    assert not np.allclose(recurrent(None,'R1',None,x,None,{9})[9],recurrent(None,'R1',None,other,None,{9})[9])

def test_observed_exemplar_still_affects_query():
    x=np.random.default_rng(10).normal(size=(2,7,7));other=x.copy();other[:,:4]+=2
    assert not np.allclose(anchor_features(TensorOps,'C',None,x,None,{6},recurrent)[6],anchor_features(TensorOps,'C',None,other,None,{6},recurrent)[6])
